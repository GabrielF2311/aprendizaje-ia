"""
ÁLGEBRA LINEAL - DÍA 6: NumPy para Álgebra Lineal
==================================================

Aprende a usar NumPy para operaciones de álgebra lineal.
"""

import numpy as np
import time

# ============================================================================
# EJERCICIO 1: Creación de Arrays con NumPy
# ============================================================================

def numpy_basics():
    """Operaciones básicas con NumPy arrays"""
    
    # TODO: Crea un array 2D de forma (3, 4) con números aleatorios
    arr = None
    
    # TODO: Crea una matriz identidad 5x5
    I = None
    
    # TODO: Crea una matriz diagonal con valores [1, 2, 3, 4]
    D = None
    
    # TODO: Crea un array con valores del 0 al 20 (shape: 4x5)
    arr_range = None


# ============================================================================
# EJERCICIO 2: Operaciones Matriciales con NumPy
# ============================================================================

def numpy_operations():
    """Operaciones de álgebra lineal con NumPy"""
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # TODO: Suma de matrices
    suma = None
    
    # TODO: Multiplicación elemento a elemento (Hadamard)
    hadamard = None
    
    # TODO: Multiplicación matricial (usa @ o np.dot)
    producto = None
    
    # TODO: Transposición
    AT = None
    
    # TODO: Inversa (si existe)
    try:
        A_inv = None
    except:
        print("Matriz singular")
    
    # TODO: Determinante
    det_A = None
    
    # TODO: Traza (suma de diagonal)
    trace_A = None


# ============================================================================
# EJERCICIO 3: Resolución de Sistemas con NumPy
# ============================================================================

def solve_with_numpy():
    """
    Resuelve Ax = b usando np.linalg.solve()
    """
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]])
    b = np.array([8, -11, -3])
    
    # TODO: Resuelve usando np.linalg.solve
    x = None
    
    # TODO: Verifica la solución: Ax debería dar b
    verification = None
    
    print(f"Solución: {x}")
    print(f"Verificación Ax = {verification}")
    print(f"b = {b}")


# ============================================================================
# EJERCICIO 4: Eigenvalores y Eigenvectores (Avance)
# ============================================================================

def eigenvalues_example():
    """
    Calcula eigenvalores y eigenvectores de una matriz.
    
    Si Av = λv, entonces λ es eigenvalor y v es eigenvector.
    """
    A = np.array([[4, -2],
                  [1, 1]])
    
    # TODO: Calcula eigenvalores y eigenvectores
    eigenvalues, eigenvectors = None, None  # np.linalg.eig(A)
    
    print(f"Eigenvalores: {eigenvalues}")
    print(f"Eigenvectores:\n{eigenvectors}")
    
    # TODO: Verifica: Av = λv para el primer eigenvector
    v1 = eigenvectors[:, 0]
    lambda1 = eigenvalues[0]
    
    Av = None  # A @ v1
    lambda_v = None  # lambda1 * v1
    
    print(f"\nVerificación:")
    print(f"Av = {Av}")
    print(f"λv = {lambda_v}")


# ============================================================================
# EJERCICIO 5: Comparación de Performance
# ============================================================================

def performance_comparison():
    """
    Compara tu implementación manual vs NumPy.
    """
    from dia_04_multiplicacion import matrix_multiply
    
    # Matrices medianas
    size = 200
    A_list = [[float(i+j) for j in range(size)] for i in range(size)]
    B_list = [[float(i-j) for j in range(size)] for i in range(size)]
    
    A_np = np.array(A_list)
    B_np = np.array(B_list)
    
    # Manual
    print("Multiplicación manual...")
    start = time.time()
    C_manual = matrix_multiply(A_list, B_list)
    time_manual = time.time() - start
    
    # NumPy
    print("Multiplicación NumPy...")
    start = time.time()
    C_numpy = A_np @ B_np
    time_numpy = time.time() - start
    
    print(f"\nResultados:")
    print(f"Manual: {time_manual:.4f}s")
    print(f"NumPy: {time_numpy:.6f}s")
    print(f"Aceleración: {time_manual/time_numpy:.0f}x")


# ============================================================================
# EJERCICIO 6: Broadcasting
# ============================================================================

def broadcasting_examples():
    """
    Ejemplos de broadcasting en NumPy.
    """
    # TODO: Suma un escalar a una matriz
    A = np.array([[1, 2, 3], [4, 5, 6]])
    A_plus_10 = None  # A + 10
    
    # TODO: Suma un vector fila a cada fila de la matriz
    row_vector = np.array([1, 2, 3])
    result = None  # A + row_vector
    
    # TODO: Normaliza cada fila (resta la media de cada fila)
    row_means = None  # A.mean(axis=1, keepdims=True)
    normalized = None  # A - row_means


if __name__ == "__main__":
    print("=" * 60)
    print("DÍA 6: NumPy para Álgebra Lineal")
    print("=" * 60)
    
    # Descomenta para ejecutar:
    # numpy_basics()
    # numpy_operations()
    # solve_with_numpy()
    # eigenvalues_example()
    # performance_comparison()
    # broadcasting_examples()
