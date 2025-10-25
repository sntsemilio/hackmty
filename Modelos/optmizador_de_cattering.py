import math
from typing import Dict, List
import numpy as np
from statistics import stdev


# =============================================================================
# MODULO 1: FORECAST AGENT (Agente de Prediccion)
# =============================================================================

def get_forecast(product_id: str, flight_id: str) -> Dict[str, float]:
    """
    Simula la prediccion de un modelo de Machine Learning (ej. XGBoost).
    
    En produccion, esta funcion llamaria a un modelo entrenado (.predict()).
    Por ahora, simula valores basados en el tipo de producto.
    
    Args:
        product_id: Identificador del producto (ej. "SNACK_001", "BEBIDA_002")
        flight_id: Identificador del vuelo (ej. "AA1234")
    
    Returns:
        Dict con dos claves:
            - p_served: Probabilidad de consumo (float entre 0.0 y 1.0)
            - quantity_predicted: Cantidad esperada de consumo (int)
    
    Example:
        >>> result = get_forecast("SNACK_001", "AA1234")
        >>> print(result)
        {'p_served': 0.85, 'quantity_predicted': 50}
    """
    # Logica de simulacion basada en el tipo de producto
    if "SNACK" in product_id.upper():
        p_served = 0.85
        quantity_predicted = 50
    elif "BEBIDA" in product_id.upper():
        p_served = 0.60
        quantity_predicted = 30
    else:
        p_served = 0.30
        quantity_predicted = 10
    
    return {
        "p_served": float(p_served),
        "quantity_predicted": int(quantity_predicted)
    }


# =============================================================================
# MODULO 2: FRESHNESS AGENT (Agente de Frescura)
# =============================================================================

def calculate_freshness_score(days_to_expiry: int, product_type: str) -> float:
    """
    Calcula un puntaje de urgencia basado en la fecha de caducidad.
    
    Formula: FreshnessScore = e^(-lambda * days_to_expiry)
    
    Donde lambda (lambda) varia segun el tipo de producto:
        - PERECEDERO: lambda = 0.3 (decaimiento rapido)
        - BEBIDA: lambda = 0.1 (decaimiento medio)
        - SNACK_SECO: lambda = 0.05 (decaimiento lento)
        - DEFAULT: lambda = 0.05
    
    Args:
        days_to_expiry: Dias restantes hasta la caducidad (int)
        product_type: Tipo de producto (str)
    
    Returns:
        Puntaje de frescura (float). Valores altos = mayor urgencia
        
    Example:
        >>> calculate_freshness_score(10, "PERECEDERO")
        0.049787068367863944
        >>> calculate_freshness_score(10, "SNACK_SECO")
        0.6065306597126334
    """
    # Mapeo de tipo de producto a constante de decaimiento
    lambda_map: Dict[str, float] = {
        "PERECEDERO": 0.3,
        "BEBIDA": 0.1,
        "SNACK_SECO": 0.05
    }
    
    # Seleccionar lambda segun el tipo de producto
    lambda_decay: float = lambda_map.get(product_type.upper(), 0.05)
    
    # Si el producto ya caduco (dias negativos), tratarlo como 0
    # Esto resulta en e^0 = 1.0 (maxima urgencia)
    effective_days: int = max(0, days_to_expiry)
    
    # Calcular el puntaje de frescura usando decaimiento exponencial
    freshness_score: float = math.exp(-lambda_decay * effective_days)
    
    return freshness_score


# =============================================================================
# MODULO 3: FUSION AGENT (Agente de Fusion)
# =============================================================================

def calculate_priority(p_served: float, freshness_score: float) -> float:
    """
    Combina la demanda (forecast) y la urgencia (freshness) en un puntaje unico.
    
    Formula: Priority = w_p * P_served + w_f * FreshnessScore
    
    Pesos:
        - w_p (peso de prediccion/demanda) = 0.55
        - w_f (peso de frescura/urgencia) = 0.45
    
    Args:
        p_served: Probabilidad de ser servido/consumido (0.0 - 1.0)
        freshness_score: Puntaje de frescura del producto
    
    Returns:
        Puntaje de prioridad combinado (float)
        
    Example:
        >>> calculate_priority(0.85, 0.60)
        0.7375
    """
    # Pesos definidos para la combinacion
    w_p: float = 0.55  # Peso de prediccion/demanda
    w_f: float = 0.45  # Peso de frescura/urgencia
    
    # Calcular prioridad como combinacion lineal ponderada
    priority: float = (w_p * p_served) + (w_f * freshness_score)
    
    return priority


def normalize_priorities(priority_scores: List[float]) -> List[float]:
    """
    Normaliza los puntajes de prioridad para que sumen 1.0 (distribucion de probabilidad).
    
    Formula: p_tilde_i = Priority_i / Sum(Priority_j) para todos los j activos
    
    Args:
        priority_scores: Lista de puntajes de prioridad sin normalizar
    
    Returns:
        Lista de probabilidades normalizadas que suman 1.0
        
    Example:
        >>> normalize_priorities([0.8, 0.6, 0.6])
        [0.4, 0.3, 0.3]
        >>> normalize_priorities([0.0, 0.0])
        [0.5, 0.5]
    """
    # Manejar lista vacia
    if not priority_scores:
        return []
    
    # Calcular suma total
    total_priority: float = sum(priority_scores)
    
    # Manejar caso de suma = 0 (evitar division por cero)
    if total_priority == 0.0:
        # Devolver distribucion uniforme
        uniform_prob: float = 1.0 / len(priority_scores)
        return [uniform_prob] * len(priority_scores)
    
    # Normalizar cada puntaje dividiendo por la suma total
    normalized: List[float] = [
        score / total_priority for score in priority_scores
    ]
    
    return normalized


# =============================================================================
# MODULO 4: ANALYTICS AGENT (Agente de Analitica)
# =============================================================================

def calculate_session_efficiency(session_json: Dict) -> float:
    """
    Extrae la metrica de eficiencia de una sesion de trabajo.
    
    Args:
        session_json: Diccionario con datos de la sesion, debe contener
                      la clave 'tasa_items_por_minuto'
    
    Returns:
        Tasa de items procesados por minuto (float)
        
    Example:
        >>> session = {
        ...     "sesion_id": "20251025_014648",
        ...     "tasa_items_por_minuto": 2.3,
        ...     "timestamps_items": [7.06, 9.28, 11.5]
        ... }
        >>> calculate_session_efficiency(session)
        2.3
    """
    return float(session_json.get("tasa_items_por_minuto", 0.0))


def calculate_session_stability(session_json: Dict) -> float:
    """
    Calcula la estabilidad de una sesion basandose en la variabilidad de timestamps.
    
    Formula: session_stability = e^(-SD(deltas))
    
    Donde deltas son los intervalos entre timestamps consecutivos.
    Un SD bajo (trabajo consistente) -> puntaje cercano a 1.0
    Un SD alto (trabajo erratico) -> puntaje cercano a 0.0
    
    Args:
        session_json: Diccionario con datos de la sesion, debe contener
                      la clave 'timestamps_items' (lista de floats)
    
    Returns:
        Puntaje de estabilidad (float entre 0.0 y 1.0)
        
    Example:
        >>> session = {
        ...     "sesion_id": "20251025_014648",
        ...     "timestamps_items": [7.06, 9.28, 11.5, 12.8, 15.0]
        ... }
        >>> calculate_session_stability(session)
        0.6977746579640468
    """
    timestamps: List[float] = session_json.get("timestamps_items", [])
    
    # Caso de borde: menos de 3 timestamps (menos de 2 intervalos)
    if len(timestamps) < 3:
        return 0.5  # Puntaje neutro cuando no hay suficientes datos
    
    # Paso A: Calcular deltas (intervalos entre timestamps consecutivos)
    deltas: List[float] = [
        timestamps[i + 1] - timestamps[i] 
        for i in range(len(timestamps) - 1)
    ]
    
    # Paso B: Calcular desviacion estandar de los deltas
    # Usar desviacion estandar de la muestra (n-1)
    try:
        std_deviation: float = stdev(deltas)
    except Exception:
        # Si hay algun error (ej. todos los deltas son iguales)
        std_deviation = 0.0
    
    # Paso C: Aplicar formula de estabilidad
    stability: float = math.exp(-std_deviation)
    
    return stability


# =============================================================================
# MODULO 5: ASSIGNMENT AGENT (Agente de Asignacion)
# =============================================================================

def calculate_assignment_score(
    efficiency_avg: float, 
    stability_avg: float, 
    experience: float
) -> float:
    """
    Calcula el puntaje final de un empleado para asignacion a tareas.
    
    Formula: AssignmentScore = (w_e * efficiency_avg) + 
                                (w_s * stability_avg) + 
                                (w_x * experience)
    
    Pesos:
        - w_e (peso de eficiencia) = 0.5
        - w_s (peso de estabilidad) = 0.3
        - w_x (peso de experiencia) = 0.2
    
    Args:
        efficiency_avg: Promedio historico de eficiencia (items/min)
        stability_avg: Promedio historico de estabilidad (0.0 - 1.0)
        experience: Años de experiencia del empleado
    
    Returns:
        Puntaje de asignacion del empleado (float)
        
    Example:
        >>> calculate_assignment_score(2.5, 0.85, 3.0)
        2.105
    """
    # Pesos definidos para la asignacion
    w_e: float = 0.5  # Peso de eficiencia
    w_s: float = 0.3  # Peso de estabilidad
    w_x: float = 0.2  # Peso de experiencia
    
    # Calcular puntaje de asignacion como combinacion lineal ponderada
    assignment_score: float = (
        (w_e * efficiency_avg) + 
        (w_s * stability_avg) + 
        (w_x * experience)
    )
    
    return assignment_score


# =============================================================================
# FUNCIONES DE UTILIDAD Y TESTING
# =============================================================================

def demo_pipeline() -> None:
    """
    Funcion de demostracion que ejecuta un flujo completo del sistema.
    """
    print("=" * 70)
    print("DEMO: Sistema Integral de Optimizacion para Airline Catering")
    print("=" * 70)
    
    # 1. FORECAST AGENT
    print("\n[1] FORECAST AGENT - Prediccion de Demanda")
    forecast = get_forecast("SNACK_001", "AA1234")
    print(f"   Producto: SNACK_001")
    print(f"   P(servido): {forecast['p_served']}")
    print(f"   Cantidad predicha: {forecast['quantity_predicted']}")
    
    # 2. FRESHNESS AGENT
    print("\n[2] FRESHNESS AGENT - Calculo de Frescura")
    freshness = calculate_freshness_score(5, "PERECEDERO")
    print(f"   Producto: PERECEDERO (5 dias para caducar)")
    print(f"   Freshness Score: {freshness:.4f}")
    
    # 3. FUSION AGENT
    print("\n[3] FUSION AGENT - Priorizacion")
    priority = calculate_priority(forecast['p_served'], freshness)
    print(f"   Priority Score: {priority:.4f}")
    
    # Normalizacion de multiples lotes
    priorities = [0.8, 0.6, 0.6]
    normalized = normalize_priorities(priorities)
    print(f"   Prioridades originales: {priorities}")
    print(f"   Prioridades normalizadas: {[f'{p:.2f}' for p in normalized]}")
    
    # 4. ANALYTICS AGENT
    print("\n[4] ANALYTICS AGENT - Analisis de Sesion")
    session_data = {
        "sesion_id": "20251025_014648",
        "tasa_items_por_minuto": 2.3,
        "timestamps_items": [7.06, 9.28, 11.5, 12.8, 15.0]
    }
    efficiency = calculate_session_efficiency(session_data)
    stability = calculate_session_stability(session_data)
    print(f"   Eficiencia: {efficiency:.2f} items/min")
    print(f"   Estabilidad: {stability:.4f}")
    
    # 5. ASSIGNMENT AGENT
    print("\n[5] ASSIGNMENT AGENT - Asignacion de Empleado")
    assignment = calculate_assignment_score(2.5, 0.85, 3.0)
    print(f"   Empleado: Eficiencia=2.5, Estabilidad=0.85, Experiencia=3 años")
    print(f"   Assignment Score: {assignment:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Ejecutar demostracion
    demo_pipeline()
    
    # Tests unitarios basicos
    print("\n[TESTS UNITARIOS]")
    
    # Test 1: Freshness con producto caducado
    assert calculate_freshness_score(-5, "PERECEDERO") == 1.0, "Producto caducado debe tener score = 1.0"
    print("[OK] Test 1: Freshness con producto caducado")
    
    # Test 2: Normalizacion suma 1.0
    normalized = normalize_priorities([0.5, 0.3, 0.2])
    assert abs(sum(normalized) - 1.0) < 0.0001, "Normalizacion debe sumar 1.0"
    print("[OK] Test 2: Normalizacion suma 1.0")
    
    # Test 3: Stability con pocos datos
    session_min = {"timestamps_items": [1.0, 2.0]}
    stability = calculate_session_stability(session_min)
    assert stability == 0.5, "Pocos timestamps deben retornar 0.5"
    print("[OK] Test 3: Stability con datos insuficientes")
    
    # Test 4: Assignment score con valores conocidos
    score = calculate_assignment_score(2.0, 1.0, 5.0)
    expected = (0.5 * 2.0) + (0.3 * 1.0) + (0.2 * 5.0)
    assert abs(score - expected) < 0.0001, "Assignment score incorrecto"
    print("[OK] Test 4: Assignment score")
    
    print("\n[PASS] Todos los tests pasaron exitosamente!")