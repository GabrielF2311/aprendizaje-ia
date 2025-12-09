"""
ÁLGEBRA LINEAL - DÍA 5: SISTEMAS DE ECUACIONES
===============================================

Resuelve sistemas lineales usando eliminación Gaussiana.
"""

import numpy as np
from typing import List, Optional, Tuple

Matrix = List[List[float]]
Vector = List[float]

# ============================================================================
# EJERCICIO 1: Eliminación Gaussiana
# ============================================================================

def gaussian_elimination(A: Matrix, b: Vector) -> Optional[Vector]:
    """
    Resuelve Ax = b usando eliminación Gaussiana.
    
    Pasos:
    1. Forma aumentada [A|b]
    2. Eliminación hacia adelante (forma escalonada)
    3. Sustitución hacia atrás
    
    Args:
        A: Matriz de coeficientes (n×n)
        b: Vector de términos independientes
        
    Returns:
        Vector solución x, o None si no hay solución única
    """
    # TODO: Implementa eliminación Gaussiana
    pass


def forward_elimination(augmented: Matrix) -> Matrix:
    """
    Convierte la matriz en forma escalonada.
    """
    # TODO: Implementa eliminación hacia adelante
    pass


def back_substitution(augmented: Matrix) -> Vector:
    """
    Resuelve el sistema usando sustitución hacia atrás.
    """
    # TODO: Implementa sustitución hacia atrás
    pass


# ============================================================================
# EJERCICIO 2: Método de Gauss-Jordan
# ============================================================================

def gauss_jordan(A: Matrix, b: Vector) -> Optional[Vector]:
    """
    Resuelve Ax = b usando Gauss-Jordan (lleva a forma reducida).
    """
    # TODO: Implementa Gauss-Jordan
    pass


# ============================================================================
# EJERCICIO 3: Casos Especiales
# ============================================================================

def solve_system_3x3():
    """
    Resuelve un sistema 3x3 específico:
    
    2x + y - z = 8
    -3x - y + 2z = -11
    -2x + y + 2z = -3
    """
    A = [[2, 1, -1],
         [-3, -1, 2],
         [-2, 1, 2]]
    b = [8, -11, -3]
    
    # TODO: Resuelve usando tus funciones
    # La solución es x=2, y=3, z=-1
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("DÍA 5: SISTEMAS DE ECUACIONES")
    print("=" * 60)
    
    # solve_system_3x3()
