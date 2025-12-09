"""
ÁLGEBRA LINEAL - DÍA 2: OPERACIONES CON VECTORES
================================================

Aprende operaciones fundamentales: suma, resta, producto punto, ángulos.
"""

import numpy as np
import math
from typing import List, Tuple

# ============================================================================
# EJERCICIO 1: Suma y Resta de Vectores
# ============================================================================

def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """
    Suma dos vectores elemento a elemento.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        Vector resultante de v1 + v2
        
    Ejemplo:
        >>> vector_add([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
    """
    # TODO: Implementa la suma
    # Pista: usa zip() para iterar sobre ambos vectores
    pass


def vector_subtract(v1: List[float], v2: List[float]) -> List[float]:
    """
    Resta dos vectores elemento a elemento.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        Vector resultante de v1 - v2
        
    Ejemplo:
        >>> vector_subtract([5, 7, 9], [1, 2, 3])
        [4, 5, 6]
    """
    # TODO: Implementa la resta
    pass


def scalar_multiply(scalar: float, vector: List[float]) -> List[float]:
    """
    Multiplica un vector por un escalar.
    
    Args:
        scalar: Número por el cual multiplicar
        vector: Vector a multiplicar
        
    Returns:
        Vector resultante
        
    Ejemplo:
        >>> scalar_multiply(2, [1, 2, 3])
        [2, 4, 6]
    """
    # TODO: Implementa la multiplicación por escalar
    pass


# ============================================================================
# EJERCICIO 2: Producto Punto (Dot Product)
# ============================================================================

def dot_product(v1: List[float], v2: List[float]) -> float:
    """
    Calcula el producto punto de dos vectores.
    
    Fórmula: v1 · v2 = v1[0]*v2[0] + v1[1]*v2[1] + ... + v1[n]*v2[n]
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        Producto punto (un número)
        
    Ejemplo:
        >>> dot_product([1, 2, 3], [4, 5, 6])
        32  # (1*4 + 2*5 + 3*6)
    """
    # TODO: Implementa el producto punto
    # Pista: suma de multiplicaciones elemento a elemento
    pass


def dot_product_numpy(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Producto punto usando NumPy.
    
    Compara la velocidad con tu implementación.
    """
    # TODO: Usa np.dot() o el operador @
    pass


# ============================================================================
# EJERCICIO 3: Ángulo entre Vectores
# ============================================================================

def angle_between_vectors(v1: List[float], v2: List[float], 
                         degrees: bool = True) -> float:
    """
    Calcula el ángulo entre dos vectores.
    
    Fórmula: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
             θ = arccos((v1 · v2) / (||v1|| * ||v2||))
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        degrees: Si True, retorna en grados; si False, en radianes
        
    Returns:
        Ángulo entre los vectores
        
    Ejemplo:
        >>> angle_between_vectors([1, 0], [0, 1])
        90.0  # Son perpendiculares
    """
    # TODO: Implementa esto
    # Pasos:
    # 1. Calcula el producto punto
    # 2. Calcula las magnitudes de ambos vectores
    # 3. Calcula cos(θ) = dot / (mag1 * mag2)
    # 4. Usa math.acos() para obtener θ
    # 5. Convierte a grados si es necesario: math.degrees()
    pass


def are_perpendicular(v1: List[float], v2: List[float], 
                     tolerance: float = 1e-10) -> bool:
    """
    Verifica si dos vectores son perpendiculares (ortogonales).
    
    Dos vectores son perpendiculares si su producto punto es 0.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        tolerance: Tolerancia para comparación
        
    Returns:
        True si son perpendiculares
        
    Ejemplo:
        >>> are_perpendicular([1, 0], [0, 1])
        True
    """
    # TODO: Implementa esto
    pass


def are_parallel(v1: List[float], v2: List[float], 
                tolerance: float = 1e-10) -> bool:
    """
    Verifica si dos vectores son paralelos.
    
    Dos vectores son paralelos si el ángulo entre ellos es 0° o 180°.
    
    Ejemplo:
        >>> are_parallel([2, 4], [1, 2])
        True  # [2, 4] = 2 * [1, 2]
    """
    # TODO: Implementa esto
    # Pista: calcula el ángulo y verifica si es ~0° o ~180°
    pass


# ============================================================================
# EJERCICIO 4: Proyección de Vectores
# ============================================================================

def project_onto(v: List[float], onto: List[float]) -> List[float]:
    """
    Proyecta el vector v sobre el vector onto.
    
    Fórmula: proj_onto(v) = ((v · onto) / (onto · onto)) * onto
    
    Args:
        v: Vector a proyectar
        onto: Vector sobre el cual proyectar
        
    Returns:
        Proyección de v sobre onto
        
    Ejemplo:
        >>> project_onto([3, 4], [1, 0])
        [3.0, 0.0]  # Proyección sobre eje X
    """
    # TODO: Implementa la proyección
    pass


def component_parallel(v: List[float], direction: List[float]) -> List[float]:
    """
    Calcula la componente de v paralela a direction.
    (Es lo mismo que la proyección)
    """
    return project_onto(v, direction)


def component_perpendicular(v: List[float], direction: List[float]) -> List[float]:
    """
    Calcula la componente de v perpendicular a direction.
    
    Fórmula: v_perp = v - proj_direction(v)
    """
    # TODO: Implementa esto
    # Resta la proyección del vector original
    pass


# ============================================================================
# EJERCICIO 5: Producto Cruz (Solo para 3D)
# ============================================================================

def cross_product_3d(v1: List[float], v2: List[float]) -> List[float]:
    """
    Calcula el producto cruz de dos vectores 3D.
    
    Fórmula: v1 × v2 = [v1[1]*v2[2] - v1[2]*v2[1],
                        v1[2]*v2[0] - v1[0]*v2[2],
                        v1[0]*v2[1] - v1[1]*v2[0]]
    
    El resultado es un vector perpendicular a ambos.
    
    Args:
        v1: Vector 3D
        v2: Vector 3D
        
    Returns:
        Vector perpendicular a v1 y v2
        
    Ejemplo:
        >>> cross_product_3d([1, 0, 0], [0, 1, 0])
        [0, 0, 1]  # Perpendicular a X y Y es Z
    """
    # TODO: Implementa el producto cruz
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Producto cruz solo funciona en 3D")
    
    # Usa la fórmula de arriba
    pass


# ============================================================================
# EJERCICIO 6: Combinaciones Lineales
# ============================================================================

def linear_combination(scalars: List[float], vectors: List[List[float]]) -> List[float]:
    """
    Calcula una combinación lineal de vectores.
    
    Fórmula: c1*v1 + c2*v2 + ... + cn*vn
    
    Args:
        scalars: Coeficientes [c1, c2, ..., cn]
        vectors: Vectores [v1, v2, ..., vn]
        
    Returns:
        Vector resultante
        
    Ejemplo:
        >>> linear_combination([2, 3], [[1, 0], [0, 1]])
        [2, 3]  # 2*[1,0] + 3*[0,1]
    """
    # TODO: Implementa esto
    # 1. Multiplica cada escalar por su vector
    # 2. Suma todos los resultados
    pass


# ============================================================================
# EJERCICIO 7: Visualización
# ============================================================================

def plot_vectors_2d(*vectors, labels=None, colors=None):
    """
    Dibuja múltiples vectores 2D.
    
    Ejemplo:
        >>> v1 = [3, 2]
        >>> v2 = [1, 3]
        >>> v_sum = vector_add(v1, v2)
        >>> plot_vectors_2d(v1, v2, v_sum, 
                           labels=['v1', 'v2', 'v1+v2'],
                           colors=['blue', 'red', 'green'])
    """
    import matplotlib.pyplot as plt
    
    if labels is None:
        labels = [f'v{i}' for i in range(len(vectors))]
    if colors is None:
        colors = ['blue'] * len(vectors)
    
    plt.figure(figsize=(8, 8))
    
    for vec, label, color in zip(vectors, labels, colors):
        if len(vec) != 2:
            continue
        plt.quiver(0, 0, vec[0], vec[1], 
                  angles='xy', scale_units='xy', scale=1,
                  color=color, label=label, width=0.006)
    
    # Configurar plot
    max_val = max(max(abs(v[0]), abs(v[1])) for v in vectors if len(v) == 2)
    limit = max_val * 1.2
    
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.title('Vectores 2D')
    plt.show()


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """Ejecuta tests para verificar tus implementaciones"""
    print("🧪 Ejecutando tests...\n")
    
    # Test 1: Suma
    print("Test 1: Suma de vectores")
    result = vector_add([1, 2, 3], [4, 5, 6])
    assert result == [5, 7, 9], f"Error: {result}"
    print("✅ Suma correcta\n")
    
    # Test 2: Producto punto
    print("Test 2: Producto punto")
    result = dot_product([1, 2, 3], [4, 5, 6])
    assert result == 32, f"Error: esperado 32, obtenido {result}"
    print("✅ Producto punto correcto\n")
    
    # Test 3: Ángulo entre vectores
    print("Test 3: Ángulo")
    angle = angle_between_vectors([1, 0], [0, 1])
    assert abs(angle - 90) < 0.01, f"Error: esperado 90°, obtenido {angle}"
    print("✅ Ángulo correcto\n")
    
    # Test 4: Perpendiculares
    print("Test 4: Vectores perpendiculares")
    assert are_perpendicular([1, 0], [0, 1]) == True
    assert are_perpendicular([1, 1], [1, 1]) == False
    print("✅ Detección de perpendicularidad correcta\n")
    
    # Test 5: Paralelos
    print("Test 5: Vectores paralelos")
    assert are_parallel([2, 4], [1, 2]) == True
    assert are_parallel([1, 0], [0, 1]) == False
    print("✅ Detección de paralelismo correcta\n")
    
    # Test 6: Producto cruz
    print("Test 6: Producto cruz")
    result = cross_product_3d([1, 0, 0], [0, 1, 0])
    assert result == [0, 0, 1], f"Error: {result}"
    print("✅ Producto cruz correcto\n")
    
    print("🎉 ¡Todos los tests pasaron!")


# ============================================================================
# DESAFÍOS
# ============================================================================

def challenge_gram_schmidt():
    """
    DESAFÍO AVANZADO: Implementa el proceso de Gram-Schmidt.
    
    Convierte un conjunto de vectores linealmente independientes
    en un conjunto de vectores ortonormales.
    
    No te preocupes si no puedes hacerlo aún, lo veremos en semanas 3-4.
    """
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ÁLGEBRA LINEAL - DÍA 2: OPERACIONES CON VECTORES")
    print("=" * 60)
    print()
    
    # Ejemplo de uso
    print("📝 Ejemplos:\n")
    
    # Suma de vectores
    v1 = [3, 4]
    v2 = [1, 2]
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    # v_sum = vector_add(v1, v2)
    # print(f"v1 + v2 = {v_sum}\n")
    
    # Producto punto
    # dot = dot_product([1, 2, 3], [4, 5, 6])
    # print(f"[1,2,3] · [4,5,6] = {dot}\n")
    
    # Ángulo
    # angle = angle_between_vectors([1, 0], [1, 1])
    # print(f"Ángulo entre [1,0] y [1,1]: {angle}°\n")
    
    # Ejecuta los tests cuando completes las funciones
    # run_tests()
    
    print("\n💡 Completa todas las funciones y ejecuta los tests!")
