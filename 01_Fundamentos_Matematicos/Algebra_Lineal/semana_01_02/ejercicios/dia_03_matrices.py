"""
ÁLGEBRA LINEAL - DÍA 3: MATRICES
=================================

Aprende a crear y manipular matrices desde cero.
"""

import numpy as np
from typing import List, Tuple

# Tipo personalizado para matrices
Matrix = List[List[float]]

# ============================================================================
# EJERCICIO 1: Creación de Matrices
# ============================================================================

def create_matrix(rows: int, cols: int, fill_value: float = 0.0) -> Matrix:
    """
    Crea una matriz de dimensiones rows × cols llena con fill_value.
    
    Args:
        rows: Número de filas
        cols: Número de columnas
        fill_value: Valor para llenar la matriz
        
    Returns:
        Matriz rows × cols
        
    Ejemplo:
        >>> create_matrix(2, 3, 0.0)
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]
    """
    # TODO: Implementa esto
    # Pista: list comprehension anidado
    pass


def identity_matrix(n: int) -> Matrix:
    """
    Crea una matriz identidad n × n.
    
    La matriz identidad tiene 1s en la diagonal y 0s en el resto.
    
    Args:
        n: Tamaño de la matriz
        
    Returns:
        Matriz identidad n × n
        
    Ejemplo:
        >>> identity_matrix(3)
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    """
    # TODO: Implementa esto
    pass


def zero_matrix(rows: int, cols: int) -> Matrix:
    """Crea una matriz de ceros"""
    # TODO: Usa create_matrix con fill_value=0
    pass


def ones_matrix(rows: int, cols: int) -> Matrix:
    """Crea una matriz de unos"""
    # TODO: Usa create_matrix con fill_value=1
    pass


def diagonal_matrix(diagonal: List[float]) -> Matrix:
    """
    Crea una matriz diagonal a partir de una lista de valores.
    
    Ejemplo:
        >>> diagonal_matrix([1, 2, 3])
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]]
    """
    # TODO: Implementa esto
    pass


# ============================================================================
# EJERCICIO 2: Propiedades de Matrices
# ============================================================================

def matrix_shape(matrix: Matrix) -> Tuple[int, int]:
    """
    Retorna las dimensiones (rows, cols) de una matriz.
    
    Ejemplo:
        >>> matrix_shape([[1, 2, 3], [4, 5, 6]])
        (2, 3)
    """
    # TODO: Implementa esto
    pass


def is_square(matrix: Matrix) -> bool:
    """
    Verifica si una matriz es cuadrada (rows == cols).
    """
    # TODO: Implementa esto
    pass


def is_symmetric(matrix: Matrix) -> bool:
    """
    Verifica si una matriz es simétrica (A == A^T).
    
    Una matriz es simétrica si A[i][j] == A[j][i] para todo i, j.
    """
    # TODO: Implementa esto
    # 1. Verifica que sea cuadrada
    # 2. Compara cada elemento con su transpuesto
    pass


def get_element(matrix: Matrix, row: int, col: int) -> float:
    """Obtiene el elemento en la posición (row, col)"""
    return matrix[row][col]


def set_element(matrix: Matrix, row: int, col: int, value: float) -> Matrix:
    """
    Establece el valor de un elemento (retorna nueva matriz, no modifica original).
    """
    # TODO: Crea una copia y modifica
    import copy
    new_matrix = copy.deepcopy(matrix)
    new_matrix[row][col] = value
    return new_matrix


# ============================================================================
# EJERCICIO 3: Operaciones Básicas
# ============================================================================

def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    """
    Suma dos matrices elemento a elemento.
    
    Las matrices deben tener las mismas dimensiones.
    
    Ejemplo:
        >>> A = [[1, 2], [3, 4]]
        >>> B = [[5, 6], [7, 8]]
        >>> matrix_add(A, B)
        [[6, 8], [10, 12]]
    """
    # TODO: Implementa la suma
    # 1. Verifica que tengan las mismas dimensiones
    # 2. Suma elemento a elemento
    pass


def matrix_subtract(A: Matrix, B: Matrix) -> Matrix:
    """Resta dos matrices elemento a elemento"""
    # TODO: Similar a la suma
    pass


def scalar_multiply_matrix(scalar: float, matrix: Matrix) -> Matrix:
    """
    Multiplica una matriz por un escalar.
    
    Ejemplo:
        >>> scalar_multiply_matrix(2, [[1, 2], [3, 4]])
        [[2, 4], [6, 8]]
    """
    # TODO: Multiplica cada elemento por el escalar
    pass


# ============================================================================
# EJERCICIO 4: Transposición
# ============================================================================

def transpose(matrix: Matrix) -> Matrix:
    """
    Transpone una matriz (intercambia filas por columnas).
    
    Ejemplo:
        >>> transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4],
         [2, 5],
         [3, 6]]
    """
    # TODO: Implementa la transposición
    # Pista: La fila i se convierte en la columna i
    pass


# ============================================================================
# EJERCICIO 5: Multiplicación de Matrices
# ============================================================================

def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """
    Multiplica dos matrices.
    
    Regla: A(m×n) × B(n×p) = C(m×p)
    C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    
    Args:
        A: Matriz m × n
        B: Matriz n × p
        
    Returns:
        Matriz m × p
        
    Ejemplo:
        >>> A = [[1, 2], [3, 4]]
        >>> B = [[5, 6], [7, 8]]
        >>> matrix_multiply(A, B)
        [[19, 22],   # [1*5+2*7, 1*6+2*8]
         [43, 50]]   # [3*5+4*7, 3*6+4*8]
    """
    # TODO: Implementa la multiplicación de matrices
    # 1. Verifica compatibilidad (cols de A == rows de B)
    # 2. Crea matriz resultado (rows de A × cols de B)
    # 3. Calcula cada elemento con la fórmula de arriba
    pass


def can_multiply(A: Matrix, B: Matrix) -> bool:
    """
    Verifica si dos matrices se pueden multiplicar.
    
    A(m×n) y B(p×q) se pueden multiplicar si n == p
    """
    # TODO: Implementa esto
    pass


# ============================================================================
# EJERCICIO 6: Vectores como Matrices
# ============================================================================

def vector_to_column_matrix(vector: List[float]) -> Matrix:
    """
    Convierte un vector en una matriz columna.
    
    Ejemplo:
        >>> vector_to_column_matrix([1, 2, 3])
        [[1],
         [2],
         [3]]
    """
    # TODO: Implementa esto
    pass


def vector_to_row_matrix(vector: List[float]) -> Matrix:
    """
    Convierte un vector en una matriz fila.
    
    Ejemplo:
        >>> vector_to_row_matrix([1, 2, 3])
        [[1, 2, 3]]
    """
    # TODO: Implementa esto
    pass


def matrix_vector_multiply(matrix: Matrix, vector: List[float]) -> List[float]:
    """
    Multiplica una matriz por un vector.
    
    Ejemplo:
        >>> A = [[1, 2], [3, 4]]
        >>> v = [5, 6]
        >>> matrix_vector_multiply(A, v)
        [17, 39]  # [1*5+2*6, 3*5+4*6]
    """
    # TODO: Implementa esto
    # Opción 1: Convierte vector a matriz columna y usa matrix_multiply
    # Opción 2: Implementa directamente
    pass


# ============================================================================
# EJERCICIO 7: Visualización y Utilidades
# ============================================================================

def print_matrix(matrix: Matrix, decimals: int = 2):
    """
    Imprime una matriz de forma legible.
    
    Ejemplo:
        >>> A = [[1.5, 2.7], [3.1, 4.9]]
        >>> print_matrix(A)
        [1.50  2.70]
        [3.10  4.90]
    """
    rows, cols = matrix_shape(matrix)
    for i in range(rows):
        row_str = "["
        for j in range(cols):
            row_str += f"{matrix[i][j]:>{decimals+3}.{decimals}f}  "
        row_str = row_str.rstrip() + "]"
        print(row_str)


def matrix_to_numpy(matrix: Matrix) -> np.ndarray:
    """Convierte tu matriz a NumPy array"""
    return np.array(matrix)


def numpy_to_matrix(arr: np.ndarray) -> Matrix:
    """Convierte NumPy array a tu formato de matriz"""
    return arr.tolist()


# ============================================================================
# EJERCICIO 8: Matrices Especiales
# ============================================================================

def upper_triangular(matrix: Matrix) -> Matrix:
    """
    Convierte una matriz en triangular superior (pone 0s debajo de la diagonal).
    
    Ejemplo:
        >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> upper_triangular(A)
        [[1, 2, 3],
         [0, 5, 6],
         [0, 0, 9]]
    """
    # TODO: Implementa esto
    pass


def lower_triangular(matrix: Matrix) -> Matrix:
    """
    Convierte una matriz en triangular inferior (pone 0s arriba de la diagonal).
    """
    # TODO: Implementa esto
    pass


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """Ejecuta tests para verificar tus implementaciones"""
    print("🧪 Ejecutando tests...\n")
    
    # Test 1: Creación
    print("Test 1: Creación de matrices")
    I = identity_matrix(3)
    assert I[0][0] == 1 and I[0][1] == 0, "Error en identity_matrix"
    print("✅ Creación correcta\n")
    
    # Test 2: Shape
    print("Test 2: Shape")
    A = [[1, 2, 3], [4, 5, 6]]
    assert matrix_shape(A) == (2, 3), "Error en matrix_shape"
    print("✅ Shape correcto\n")
    
    # Test 3: Transposición
    print("Test 3: Transposición")
    A = [[1, 2], [3, 4]]
    AT = transpose(A)
    assert AT == [[1, 3], [2, 4]], f"Error: {AT}"
    print("✅ Transposición correcta\n")
    
    # Test 4: Suma
    print("Test 4: Suma de matrices")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = matrix_add(A, B)
    assert C == [[6, 8], [10, 12]], f"Error: {C}"
    print("✅ Suma correcta\n")
    
    # Test 5: Multiplicación
    print("Test 5: Multiplicación de matrices")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = matrix_multiply(A, B)
    expected = [[19, 22], [43, 50]]
    assert C == expected, f"Error: esperado {expected}, obtenido {C}"
    print("✅ Multiplicación correcta\n")
    
    print("🎉 ¡Todos los tests pasaron!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ÁLGEBRA LINEAL - DÍA 3: MATRICES")
    print("=" * 60)
    print()
    
    # Ejemplo de uso
    print("📝 Ejemplos:\n")
    
    # Crear matriz identidad
    # I = identity_matrix(3)
    # print("Matriz identidad 3x3:")
    # print_matrix(I)
    # print()
    
    # Multiplicación
    # A = [[1, 2], [3, 4]]
    # B = [[5, 6], [7, 8]]
    # C = matrix_multiply(A, B)
    # print("A × B =")
    # print_matrix(C)
    
    # Ejecuta los tests
    # run_tests()
    
    print("\n💡 Implementa todas las funciones y ejecuta los tests!")
