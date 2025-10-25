import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque
import time


class EstadoMano(Enum):
    FUERA = 0
    EN_ESTANTE = 1
    EN_CARRITO = 2


@dataclass
class MetricasCompletas:
    sesion_id: str
    fecha_inicio: str
    duracion_sesion_seg: float
    conteo_total_items: int
    tasa_items_por_minuto: float
    fps_promedio: float
    frames_procesados: int
    fuente_video: str
    timestamps_items: List[float] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
    
    def guardar(self, ruta: str = None) -> str:
        if ruta is None:
            ruta = f"sesion_{self.sesion_id}.json"
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return ruta


class ZonaROI:
    def __init__(self, nombre: str, puntos: List[Tuple[int, int]], color: Tuple[int, int, int]):
        self.nombre = nombre
        self.puntos = np.array(puntos, dtype=np.int32)
        self.color = color
        self.centroide = np.mean(self.puntos, axis=0).astype(int)
    
    def punto_dentro(self, x: int, y: int, margen: int = 0) -> bool:
        if margen > 0:
            puntos_exp = []
            for punto in self.puntos:
                vector = punto - self.centroide
                norma = np.linalg.norm(vector)
                if norma > 0:
                    puntos_exp.append(punto + (vector / norma) * margen)
            puntos_test = np.array(puntos_exp, dtype=np.int32)
        else:
            puntos_test = self.puntos
        
        resultado = cv2.pointPolygonTest(puntos_test, (float(x), float(y)), False)
        return resultado >= 0
    
    def dibujar(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.puntos], self.color)
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        cv2.polylines(frame, [self.puntos], True, self.color, 4, cv2.LINE_AA)
        
        (tw, th), _ = cv2.getTextSize(self.nombre, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, 
                     (self.centroide[0] - 10, self.centroide[1] - th - 10),
                     (self.centroide[0] + tw + 10, self.centroide[1] + 10),
                     (0, 0, 0), -1)
        cv2.putText(frame, self.nombre, tuple(self.centroide), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        return frame


class SistemaFuncionalEstable:
    KEYPOINT_HOMBRO_IZQ = 5
    KEYPOINT_HOMBRO_DER = 6
    KEYPOINT_CODO_IZQ = 7
    KEYPOINT_CODO_DER = 8
    KEYPOINT_MUNECA_IZQ = 9
    KEYPOINT_MUNECA_DER = 10
    
    CONFIANZA_MIN_PERSONA = 0.45
    CONFIANZA_MIN_POSE = 0.35
    COOLDOWN_CONTEO = 1.0
    MARGEN_ZONA = 30
    
    def __init__(self, 
                 fuente_video: Union[int, str],
                 zona_estante: List[Tuple[int, int]],
                 zona_carrito: List[Tuple[int, int]]):
        
        self.fuente_video = fuente_video
        self.sesion_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("="*80)
        print("SISTEMA FUNCIONAL ESTABLE")
        print("="*80)
        print(f"Sesión: {self.sesion_id}\n")
        
        print("Cargando modelos...")
        self.modelo_deteccion = YOLO('yolov8n.pt')
        self.modelo_pose = YOLO('yolov8n-pose.pt')
        
        self.zona_estante = ZonaROI("ESTANTE", zona_estante, (0, 255, 0))
        self.zona_carrito = ZonaROI("CARRITO", zona_carrito, (0, 165, 255))
        
        self.estado_actual = EstadoMano.FUERA
        self.historial_estados = deque(maxlen=50)
        
        self.contador_items = 0
        self.timestamps_items = []
        
        self.tiempo_inicio_sesion = None
        self.sesion_activa = True
        self.pausado = False
        self.mostrar_ayuda = True
        self.mostrar_debug = False
        
        self.ultimo_conteo_timestamp = 0
        self.frames_procesados = 0
        self.fps_history = deque(maxlen=30)
        self.frames_sin_persona = 0
        
        print("Listo\n")
    
    def obtener_munecas(self, frame: np.ndarray, bbox: List[int]) -> Optional[Dict]:
        x1, y1, x2, y2 = map(int, bbox)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        res = self.modelo_pose(crop, verbose=False)
        
        if len(res) == 0 or res[0].keypoints is None:
            return None
        
        kp_xy = res[0].keypoints.xy[0].cpu().numpy()
        kp_conf = res[0].keypoints.conf[0].cpu().numpy()
        
        if len(kp_xy) < 11:
            return None
        
        munecas = {
            'izq': {
                'pos': (int(kp_xy[self.KEYPOINT_MUNECA_IZQ][0] + x1),
                       int(kp_xy[self.KEYPOINT_MUNECA_IZQ][1] + y1)),
                'conf': float(kp_conf[self.KEYPOINT_MUNECA_IZQ])
            },
            'der': {
                'pos': (int(kp_xy[self.KEYPOINT_MUNECA_DER][0] + x1),
                       int(kp_xy[self.KEYPOINT_MUNECA_DER][1] + y1)),
                'conf': float(kp_conf[self.KEYPOINT_MUNECA_DER])
            },
            'izq_codo': {
                'pos': (int(kp_xy[self.KEYPOINT_CODO_IZQ][0] + x1),
                       int(kp_xy[self.KEYPOINT_CODO_IZQ][1] + y1)),
                'conf': float(kp_conf[self.KEYPOINT_CODO_IZQ])
            },
            'der_codo': {
                'pos': (int(kp_xy[self.KEYPOINT_CODO_DER][0] + x1),
                       int(kp_xy[self.KEYPOINT_CODO_DER][1] + y1)),
                'conf': float(kp_conf[self.KEYPOINT_CODO_DER])
            },
            'izq_hombro': {
                'pos': (int(kp_xy[self.KEYPOINT_HOMBRO_IZQ][0] + x1),
                       int(kp_xy[self.KEYPOINT_HOMBRO_IZQ][1] + y1)),
                'conf': float(kp_conf[self.KEYPOINT_HOMBRO_IZQ])
            },
            'der_hombro': {
                'pos': (int(kp_xy[self.KEYPOINT_HOMBRO_DER][0] + x1),
                       int(kp_xy[self.KEYPOINT_HOMBRO_DER][1] + y1)),
                'conf': float(kp_conf[self.KEYPOINT_HOMBRO_DER])
            }
        }
        
        if munecas['izq']['conf'] < self.CONFIANZA_MIN_POSE and \
           munecas['der']['conf'] < self.CONFIANZA_MIN_POSE:
            return None
        
        return munecas
    
    def actualizar_estado_y_contar(self, munecas: Dict, timestamp: float) -> bool:
        if self.pausado:
            return False
        
        if timestamp - self.ultimo_conteo_timestamp < self.COOLDOWN_CONTEO:
            return False
        
        if munecas['izq']['conf'] >= munecas['der']['conf']:
            muneca_activa = munecas['izq']
            lado = "IZQ"
        else:
            muneca_activa = munecas['der']
            lado = "DER"
        
        pos_x, pos_y = muneca_activa['pos']
        confianza = muneca_activa['conf']
        
        en_estante = self.zona_estante.punto_dentro(pos_x, pos_y, margen=self.MARGEN_ZONA)
        en_carrito = self.zona_carrito.punto_dentro(pos_x, pos_y, margen=self.MARGEN_ZONA)
        
        if self.mostrar_debug:
            print(f"[{timestamp:.1f}s] {lado}: ({pos_x},{pos_y}) | "
                  f"Est:{en_estante} Car:{en_carrito} | "
                  f"Estado:{self.estado_actual.name}")
        
        item_contado = False
        
        if self.estado_actual == EstadoMano.FUERA:
            if en_estante:
                self.estado_actual = EstadoMano.EN_ESTANTE
                print(f"[{timestamp:.1f}s] ✓ FUERA -> EN_ESTANTE ({lado})")
        
        elif self.estado_actual == EstadoMano.EN_ESTANTE:
            if en_carrito:
                self.estado_actual = EstadoMano.EN_CARRITO
                self.contador_items += 1
                self.timestamps_items.append(timestamp)
                self.ultimo_conteo_timestamp = timestamp
                item_contado = True
                
                print(f"\n{'='*80}")
                print(f"  ✓✓✓ ITEM #{self.contador_items} CONTADO ✓✓✓")
                print(f"  Tiempo: {timestamp:.1f}s")
                print(f"  Mano: {lado} | Confianza: {confianza:.2f}")
                print(f"{'='*80}\n")
                
            elif not en_estante:
                self.estado_actual = EstadoMano.FUERA
                print(f"[{timestamp:.1f}s] ⚠ EN_ESTANTE -> FUERA")
        
        elif self.estado_actual == EstadoMano.EN_CARRITO:
            if not en_carrito:
                self.estado_actual = EstadoMano.FUERA
                print(f"[{timestamp:.1f}s] ✓ EN_CARRITO -> FUERA")
        
        self.historial_estados.append({
            'timestamp': timestamp,
            'estado': self.estado_actual.name,
            'pos': (pos_x, pos_y),
            'en_estante': en_estante,
            'en_carrito': en_carrito
        })
        
        return item_contado
    
    def iniciar_analisis(self) -> Optional[MetricasCompletas]:
        print("Abriendo cámara...")
        cap = cv2.VideoCapture(self.fuente_video)
        
        if not cap.isOpened():
            raise ValueError(f"Error: {self.fuente_video}")
        
        # Configurar propiedades de la cámara para estabilidad
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {ancho}x{altura}")
        
        ventana = 'Sistema Estable - Q para salir'
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ventana, 1400, 900)
        
        # CRÍTICO: Configurar para que NO se cierre automáticamente
        cv2.setWindowProperty(ventana, cv2.WND_PROP_TOPMOST, 0)
        
        print("\n" + "="*80)
        print("SESIÓN INICIADA - CONTROLES:")
        print("  Q: Finalizar | R: Reset | P: Pausa | H: Ayuda | D: Debug")
        print("="*80 + "\n")
        
        self.tiempo_inicio_sesion = time.time()
        t_ultimo = time.time()
        
        frames_consecutivos_sin_leer = 0
        MAX_FRAMES_SIN_LEER = 100
        
        try:
            while self.sesion_activa:
                ret, frame = cap.read()
                
                if not ret:
                    frames_consecutivos_sin_leer += 1
                    if frames_consecutivos_sin_leer >= MAX_FRAMES_SIN_LEER:
                        print(f"\n⚠ Error: No se pueden leer frames ({frames_consecutivos_sin_leer} intentos)")
                        break
                    time.sleep(0.01)
                    continue
                
                frames_consecutivos_sin_leer = 0
                
                tiempo = time.time() - self.tiempo_inicio_sesion
                t_frame = time.time()
                
                # Procesar
                try:
                    frame_display = self._procesar(frame, tiempo)
                except Exception as e:
                    print(f"⚠ Error procesando frame: {e}")
                    frame_display = frame.copy()
                
                # FPS
                fps = 1.0 / (t_frame - t_ultimo) if (t_frame - t_ultimo) > 0 else 0
                self.fps_history.append(fps)
                t_ultimo = t_frame
                
                fps_avg = np.mean(self.fps_history) if self.fps_history else 0
                
                # HUD
                self._dibujar_hud(frame_display, tiempo, fps_avg)
                
                if self.mostrar_ayuda:
                    self._dibujar_ayuda(frame_display)
                
                # Mostrar
                try:
                    cv2.imshow(ventana, frame_display)
                except cv2.error as e:
                    print(f"⚠ Error mostrando frame: {e}")
                    break
                
                self.frames_procesados += 1
                
                # Controles - CAMBIO CRÍTICO: waitKey con timeout mayor
                key = cv2.waitKey(10) & 0xFF  # 10ms en lugar de 1ms para mayor estabilidad
                
                if key == ord('q'):
                    print("\n✓ Finalizando por petición del usuario...")
                    self.sesion_activa = False
                elif key == ord('h'):
                    self.mostrar_ayuda = not self.mostrar_ayuda
                elif key == ord('d'):
                    self.mostrar_debug = not self.mostrar_debug
                    print(f"\nDebug: {'ON' if self.mostrar_debug else 'OFF'}")
                elif key == ord('r'):
                    self._reiniciar()
                elif key == ord('p'):
                    self.pausado = not self.pausado
                    print(f"\n{'⏸ PAUSA' if self.pausado else '▶ PLAY'}")
                elif key == 27:  # ESC
                    print("\n⚠ Saliendo sin guardar...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
        
        except KeyboardInterrupt:
            print("\n\n⚠ Ctrl+C detectado")
        
        except Exception as e:
            print(f"\n⚠ Error inesperado: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            duracion = time.time() - self.tiempo_inicio_sesion
            
            # Limpiar recursos
            try:
                cap.release()
            except:
                pass
            
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            # Esperar a que se cierren las ventanas
            for _ in range(10):
                cv2.waitKey(1)
            
            print("\n" + "="*80)
            print("SESIÓN FINALIZADA")
            print("="*80)
            print(f"Duración: {duracion:.2f}s")
            print(f"Items: {self.contador_items}")
            print(f"Frames: {self.frames_procesados}")
            print("="*80)
            
            return self._calcular_metricas(duracion)
    
    def _procesar(self, frame: np.ndarray, tiempo: float) -> np.ndarray:
        out = frame.copy()
        h, w = frame.shape[:2]
        
        out = self.zona_estante.dibujar(out, 0.20)
        out = self.zona_carrito.dibujar(out, 0.20)
        
        res = self.modelo_deteccion(frame, classes=[0], verbose=False)
        
        if len(res) > 0 and len(res[0].boxes) > 0:
            boxes = res[0].boxes
            confs = boxes.conf.cpu().numpy()
            
            if len(confs) > 0 and confs[0] > self.CONFIANZA_MIN_PERSONA:
                bbox = boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                munecas = self.obtener_munecas(frame, bbox)
                
                if munecas:
                    self._dibujar_brazos(out, munecas)
                    
                    if not self.pausado:
                        if self.actualizar_estado_y_contar(munecas, tiempo):
                            self._mostrar_feedback(out, w, h)
        
        if self.pausado:
            cv2.rectangle(out, (w//2-150, 30), (w//2+150, 100), (0,0,255), -1)
            cv2.putText(out, "PAUSA", (w//2-100, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 4)
        
        return out
    
    def _dibujar_brazos(self, frame: np.ndarray, munecas: Dict) -> None:
        # Izquierdo
        if munecas['izq']['conf'] > self.CONFIANZA_MIN_POSE:
            color = (0, 255, 255)
            
            if munecas['izq_hombro']['conf'] > self.CONFIANZA_MIN_POSE and \
               munecas['izq_codo']['conf'] > self.CONFIANZA_MIN_POSE:
                cv2.line(frame, munecas['izq_hombro']['pos'], munecas['izq_codo']['pos'], 
                        color, 5, cv2.LINE_AA)
            
            if munecas['izq_codo']['conf'] > self.CONFIANZA_MIN_POSE:
                cv2.line(frame, munecas['izq_codo']['pos'], munecas['izq']['pos'], 
                        color, 5, cv2.LINE_AA)
            
            cv2.circle(frame, munecas['izq']['pos'], 14, color, -1)
            cv2.circle(frame, munecas['izq']['pos'], 16, (0, 0, 0), 2)
            
            cv2.putText(frame, "IZQ", 
                       (munecas['izq']['pos'][0] + 20, munecas['izq']['pos'][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        
        # Derecho
        if munecas['der']['conf'] > self.CONFIANZA_MIN_POSE:
            color = (255, 255, 0)
            
            if munecas['der_hombro']['conf'] > self.CONFIANZA_MIN_POSE and \
               munecas['der_codo']['conf'] > self.CONFIANZA_MIN_POSE:
                cv2.line(frame, munecas['der_hombro']['pos'], munecas['der_codo']['pos'], 
                        color, 5, cv2.LINE_AA)
            
            if munecas['der_codo']['conf'] > self.CONFIANZA_MIN_POSE:
                cv2.line(frame, munecas['der_codo']['pos'], munecas['der']['pos'], 
                        color, 5, cv2.LINE_AA)
            
            cv2.circle(frame, munecas['der']['pos'], 14, color, -1)
            cv2.circle(frame, munecas['der']['pos'], 16, (0, 0, 0), 2)
            
            cv2.putText(frame, "DER", 
                       (munecas['der']['pos'][0] + 20, munecas['der']['pos'][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
    
    def _mostrar_feedback(self, frame: np.ndarray, w: int, h: int) -> None:
        cv2.rectangle(frame, (w//2-250, h//2-100), (w//2+250, h//2+100), (0,255,0), -1)
        cv2.rectangle(frame, (w//2-250, h//2-100), (w//2+250, h//2+100), (0,200,0), 8)
        cv2.putText(frame, f"+1", (w//2-100, h//2+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,0,0), 12)
    
    def _dibujar_hud(self, frame: np.ndarray, tiempo: float, fps: float) -> None:
        h, w = frame.shape[:2]
        px, py = w - 515, 15
        pw, ph = 500, 250
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0,255,255), 4)
        
        mins = int(tiempo // 60)
        segs = int(tiempo % 60)
        tasa = (self.contador_items / (tiempo / 60)) if tiempo > 0 else 0.0
        
        y = py + 35
        info = [
            ("ESTABLE", (0,255,255), 0.75),
            ("", (0,0,0), 0.1),
            (f"Items: {self.contador_items}", (0,255,0), 1.5),
            (f"Estado: {self.estado_actual.name}", self._color_estado(), 0.75),
            ("", (0,0,0), 0.1),
            (f"Tiempo: {mins:02d}:{segs:02d}", (255,255,255), 0.70),
            (f"Tasa: {tasa:.2f}/min", (255,255,255), 0.70),
            (f"FPS: {fps:.1f}", (200,200,200), 0.65),
        ]
        
        for txt, col, esc in info:
            if txt:
                cv2.putText(frame, txt, (px+20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, esc, col, 2)
            y += 30
    
    def _dibujar_ayuda(self, frame: np.ndarray) -> None:
        px, py = 15, 15
        pw, ph = 380, 180
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (50,50,50), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255,255,0), 4)
        
        y = py + 35
        ayuda = [
            ("CONTROLES", (255,255,0), 0.85),
            ("", (0,0,0), 0.1),
            ("Q - Salir", (255,255,255), 0.65),
            ("R - Reset", (255,255,255), 0.65),
            ("P - Pausa", (255,255,255), 0.65),
            ("D - Debug", (255,255,255), 0.65),
            ("H - Ayuda", (255,255,255), 0.65),
        ]
        
        for txt, col, esc in ayuda:
            if txt:
                cv2.putText(frame, txt, (px+18, y),
                           cv2.FONT_HERSHEY_SIMPLEX, esc, col, 2)
            y += 26
    
    def _color_estado(self) -> Tuple[int, int, int]:
        return {
            EstadoMano.FUERA: (128,128,128),
            EstadoMano.EN_ESTANTE: (0,255,0),
            EstadoMano.EN_CARRITO: (0,165,255),
        }.get(self.estado_actual, (255,255,255))
    
    def _reiniciar(self) -> None:
        print("\nRESET")
        self.contador_items = 0
        self.timestamps_items = []
        self.estado_actual = EstadoMano.FUERA
        self.historial_estados.clear()
        self.ultimo_conteo_timestamp = 0
    
    def _calcular_metricas(self, dur: float) -> MetricasCompletas:
        tasa = (self.contador_items / (dur / 60)) if dur > 0 else 0.0
        
        return MetricasCompletas(
            sesion_id=self.sesion_id,
            fecha_inicio=datetime.fromtimestamp(self.tiempo_inicio_sesion).isoformat(),
            duracion_sesion_seg=round(dur, 2),
            conteo_total_items=self.contador_items,
            tasa_items_por_minuto=round(tasa, 2),
            fps_promedio=round(np.mean(self.fps_history), 2) if self.fps_history else 0.0,
            frames_procesados=self.frames_procesados,
            fuente_video=f"webcam_{self.fuente_video}" if isinstance(self.fuente_video, int) else "ip",
            timestamps_items=[round(t, 2) for t in self.timestamps_items]
        )


def selector_roi_doble(fuente: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    cap = cv2.VideoCapture(fuente)
    
    if not cap.isOpened():
        raise ValueError(f"Error: webcam {fuente}")
    
    print("Capturando...")
    ret, frame = None, None
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.1)
    
    cap.release()
    
    if not ret:
        raise ValueError("Error")
    
    h, w = frame.shape[:2]
    print(f"Resolución: {w}x{h}")
    
    ventana = 'Selector'
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
    
    pts = []
    zonas = []
    nombre = "ESTANTE"
    colores = [(0,255,0), (0,165,255)]
    idx = 0
    mouse = (0, 0)
    
    def callback(event, x, y, flags, param):
        nonlocal pts, zonas, idx, nombre, mouse
        
        mouse = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            print(f"[{nombre}] ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) >= 3:
                zonas.append(pts.copy())
                print(f"\n✓ {nombre}\n")
                pts = []
                idx += 1
                if idx < 2:
                    nombre = "CARRITO"
    
    cv2.setMouseCallback(ventana, callback)
    
    print("\nClick IZQ: Punto | Click DER: OK | Q: Continuar\n")
    
    while len(zonas) < 2:
        disp = frame.copy()
        
        for i, z in enumerate(zonas):
            p = np.array(z, dtype=np.int32)
            ovr = disp.copy()
            cv2.fillPoly(ovr, [p], colores[i])
            disp = cv2.addWeighted(disp, 0.7, ovr, 0.3, 0)
            cv2.polylines(disp, [p], True, colores[i], 4)
        
        if pts:
            for i, pt in enumerate(pts):
                cv2.circle(disp, pt, 8, colores[idx], -1)
                if i > 0:
                    cv2.line(disp, pts[i-1], pt, colores[idx], 3)
            
            if len(pts) > 2:
                cv2.line(disp, pts[-1], pts[0], colores[idx], 1)
        
        cv2.rectangle(disp, (10,10), (450,90), (0,0,0), -1)
        cv2.rectangle(disp, (10,10), (450,90), colores[idx], 4)
        
        cv2.putText(disp, f"{nombre}", (25,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, colores[idx], 2)
        cv2.putText(disp, f"Pts: {len(pts)}", (25,75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.line(disp, (mouse[0]-20, mouse[1]), (mouse[0]+20, mouse[1]), (255,255,255), 2)
        cv2.line(disp, (mouse[0], mouse[1]-20), (mouse[0], mouse[1]+20), (255,255,255), 2)
        cv2.circle(disp, mouse, 4, (0,255,0), -1)
        
        cv2.imshow(ventana, disp)
        
        if cv2.waitKey(1) & 0xFF == ord('q') and len(zonas) == 2:
            break
    
    cv2.destroyAllWindows()
    
    if len(zonas) < 2:
        raise ValueError("Faltan zonas")
    
    print(f"\n✓ ESTANTE: {zonas[0]}")
    print(f"✓ CARRITO: {zonas[1]}\n")
    
    return zonas[0], zonas[1]


def main():
    print("\n" + "="*80)
    print("SISTEMA ESTABLE v1.0 - NO SE CIERRA SOLO")
    print("="*80)
    print(f"Usuario: sntsemilio")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*80 + "\n")
    
    try:
        estante, carrito = selector_roi_doble(0)
        
        sistema = SistemaFuncionalEstable(
            fuente_video=0,
            zona_estante=estante,
            zona_carrito=carrito
        )
        
        metricas = sistema.iniciar_analisis()
        
        if metricas:
            print("\n" + "="*80)
            print("RESULTADOS")
            print("="*80)
            print(metricas.to_json())
            print("="*80 + "\n")
            
            ruta = metricas.guardar()
            print(f"✓ Guardado: {ruta}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()