"""
ÁLGEBRA LINEAL - DÍA 4: MULTIPLICACIÓN DE MATRICES
===================================================

Domina la multiplicación matricial y sus propiedades.
"""

import numpy as np
from typing import List

Matrix = List[List[float]]

# ============================================================================
# EJERCICIO 1: Multiplicación de Matrices
# ============================================================================

def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """
    Multiplica dos matrices usando el algoritmo estándar O(n³).
    
    Args:
        A: Matriz m × n
        B: Matriz n × p
        
    Returns:
        Matriz C de dimensiones m × p
    """
    # TODO: Implementa multiplicación de matrices
    pass


def verify_multiply_properties(A: Matrix, B: Matrix, C: Matrix):
    """
    Verifica las propiedades de la multiplicación:
    1. Asociatividad: (AB)C = A(BC)
    2. Distributividad: A(B+C) = AB + AC
    3. NO conmutatividad: AB ≠ BA (en general)
    """
    # TODO: Implementa verificaciones
    pass


# ============================================================================
# EJERCICIO 2: Potencias de Matrices
# ============================================================================

def matrix_power(A: Matrix, n: int) -> Matrix:
    """
    Calcula A^n (A multiplicada n veces por sí misma).
    
    Ejemplo:
        >>> A = [[1, 1], [1, 0]]
        >>> matrix_power(A, 5)  # Calcula Fibonacci!
    """
    # TODO: Implementa potencias de matrices
    pass


# ============================================================================
# EJERCICIO 3: Comparación de Velocidad
# ============================================================================

def compare_speed():
    """
    Compara la velocidad de tu implementación vs NumPy.
    """
    import time
    
    # Crear matrices grandes
    size = 100
    A = [[i+j for j in range(size)] for i in range(size)]
    B = [[i-j for j in range(size)] for i in range(size)]
    
    # Tu implementación
    start = time.time()
    C_manual = matrix_multiply(A, B)
    time_manual = time.time() - start
    
    # NumPy
    A_np = np.array(A)
    B_np = np.array(B)
    start = time.time()
    C_numpy = A_np @ B_np
    time_numpy = time.time() - start
    
    print(f"Tu implementación: {time_manual:.4f}s")
    print(f"NumPy: {time_numpy:.4f}s")
    print(f"NumPy es {time_manual/time_numpy:.1f}x más rápido")


if __name__ == "__main__":
    print("=" * 60)
    print("DÍA 4: MULTIPLICACIÓN DE MATRICES")
    print("=" * 60)
    
    # compare_speed()
