"""
NUMPY - DÍA 1: ARRAYS BÁSICOS
==============================

Aprende a crear y manipular arrays de NumPy desde cero.
"""

import numpy as np

# ============================================================================
# EJERCICIO 1: Creación de Arrays
# ============================================================================

def crear_arrays_basicos():
    """
    Crea diferentes tipos de arrays y explora sus propiedades.
    """
    print("=" * 60)
    print("EJERCICIO 1: Creación de Arrays")
    print("=" * 60)
    
    # TODO: Crea un array 1D con los números del 1 al 10
    arr_1d = None
    
    # TODO: Crea un array 2D (matriz 3x3) con números del 1 al 9
    arr_2d = None
    
    # TODO: Crea un array 3D de forma (2, 3, 4) con ceros
    arr_3d = None
    
    # Verifica tus respuestas (descomenta cuando completes)
    # print(f"1D: {arr_1d}")
    # print(f"2D:\n{arr_2d}")
    # print(f"3D shape: {arr_3d.shape}")


# ============================================================================
# EJERCICIO 2: Atributos de Arrays
# ============================================================================

def explorar_atributos():
    """
    Explora los atributos fundamentales de arrays.
    """
    # Array de ejemplo
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    
    # TODO: Imprime el shape (forma) del array
    # print(f"Shape: {arr.shape}")
    
    # TODO: Imprime el número de dimensiones (ndim)
    # print(f"Dimensiones: {arr.ndim}")
    
    # TODO: Imprime el tipo de datos (dtype)
    # print(f"Tipo de datos: {arr.dtype}")
    
    # TODO: Imprime el número total de elementos (size)
    # print(f"Total elementos: {arr.size}")
    
    # DESAFÍO: Cambia el tipo de datos a float64
    arr_float = None
    # print(f"Como float: {arr_float}")


# ============================================================================
# EJERCICIO 3: Arrays Especiales
# ============================================================================

def crear_arrays_especiales():
    """
    Crea arrays usando funciones especiales de NumPy.
    """
    # TODO: Array de ceros de forma (3, 4)
    zeros = None
    
    # TODO: Array de unos de forma (2, 3, 2)
    ones = None
    
    # TODO: Matriz identidad 5x5
    identity = None
    
    # TODO: Array con valores del 0 al 100 con paso de 10
    arange_arr = None
    
    # TODO: Array con 5 valores equiespaciados entre 0 y 1
    linspace_arr = None
    
    # TODO: Array 3x3 con números aleatorios entre 0 y 1
    random_arr = None
    
    # TODO: Array 2x2 con números aleatorios enteros entre 1 y 10
    random_int = None
    
    # Verifica (descomenta)
    # print(f"Zeros:\n{zeros}")
    # print(f"Identity:\n{identity}")
    # print(f"Linspace: {linspace_arr}")


# ============================================================================
# EJERCICIO 4: Reshape y Flatten
# ============================================================================

def manipular_formas():
    """
    Cambia la forma de arrays sin cambiar los datos.
    """
    # Array inicial
    arr = np.arange(12)  # [0, 1, 2, ..., 11]
    
    # TODO: Reshape a 3x4
    arr_3x4 = None
    
    # TODO: Reshape a 2x6
    arr_2x6 = None
    
    # TODO: Reshape a 2x2x3
    arr_3d = None
    
    # TODO: Flatten (aplanar) el array 3D a 1D
    arr_flat = None
    
    # DESAFÍO: Usa -1 para dejar que NumPy calcule una dimensión
    # Reshape arr a forma (?, 3) donde ? se calcula automáticamente
    arr_auto = None
    
    # Verifica
    # print(f"Original: {arr.shape}")
    # print(f"3x4:\n{arr_3x4}")
    # print(f"3D: {arr_3d.shape}")


# ============================================================================
# EJERCICIO 5: Operaciones Básicas
# ============================================================================

def operaciones_basicas():
    """
    Realiza operaciones aritméticas con arrays.
    """
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    
    # TODO: Suma elemento a elemento
    suma = None
    
    # TODO: Resta
    resta = None
    
    # TODO: Multiplicación elemento a elemento
    mult = None
    
    # TODO: División
    div = None
    
    # TODO: Potencia (cada elemento al cuadrado)
    cuadrado = None
    
    # TODO: Raíz cuadrada
    raiz = None
    
    # Verifica
    # print(f"a + b = {suma}")
    # print(f"a * b = {mult}")
    # print(f"a² = {cuadrado}")


# ============================================================================
# EJERCICIO 6: Agregaciones
# ============================================================================

def agregaciones():
    """
    Calcula estadísticas de arrays.
    """
    data = np.array([[1, 2, 3], 
                     [4, 5, 6], 
                     [7, 8, 9]])
    
    # TODO: Suma de todos los elementos
    suma_total = None
    
    # TODO: Media (promedio)
    media = None
    
    # TODO: Desviación estándar
    std = None
    
    # TODO: Valor mínimo
    minimo = None
    
    # TODO: Valor máximo
    maximo = None
    
    # TODO: Suma por columnas (axis=0)
    suma_cols = None
    
    # TODO: Media por filas (axis=1)
    media_filas = None
    
    # Verifica
    # print(f"Suma total: {suma_total}")
    # print(f"Media: {media}")
    # print(f"Suma por columnas: {suma_cols}")


# ============================================================================
# EJERCICIO 7: Comparaciones y Máscaras
# ============================================================================

def comparaciones():
    """
    Usa operaciones booleanas con arrays.
    """
    arr = np.array([1, 5, 10, 15, 20, 25, 30])
    
    # TODO: Máscara booleana para elementos > 15
    mask_mayor = None
    
    # TODO: Extrae elementos mayores a 15 usando la máscara
    elementos_mayores = None
    
    # TODO: Cuenta cuántos elementos son mayores a 15
    count = None
    
    # TODO: Reemplaza todos los valores > 20 con 20
    arr_capped = arr.copy()
    # arr_capped[arr_capped > 20] = 20
    
    # Verifica
    # print(f"Máscara: {mask_mayor}")
    # print(f"Mayores a 15: {elementos_mayores}")
    # print(f"Cantidad: {count}")


# ============================================================================
# EJERCICIO 8: Vectorización vs Loops
# ============================================================================

def comparar_velocidad():
    """
    Compara la velocidad de NumPy vs Python puro.
    """
    import time
    
    # Datos
    size = 1_000_000
    lista = list(range(size))
    arr = np.arange(size)
    
    # Python puro
    start = time.time()
    resultado_lista = [x ** 2 for x in lista]
    tiempo_python = time.time() - start
    
    # NumPy
    start = time.time()
    resultado_numpy = arr ** 2
    tiempo_numpy = time.time() - start
    
    print(f"Python puro: {tiempo_python:.4f}s")
    print(f"NumPy: {tiempo_numpy:.4f}s")
    print(f"NumPy es {tiempo_python/tiempo_numpy:.1f}x más rápido!")


# ============================================================================
# DESAFÍO FINAL
# ============================================================================

def desafio_matriz_especial():
    """
    DESAFÍO: Crea una matriz 5x5 con:
    - 1s en la diagonal principal
    - 2s en las diagonales secundarias
    - 0s en el resto
    
    Ejemplo para 3x3:
    [[1, 2, 0],
     [2, 1, 2],
     [0, 2, 1]]
    """
    # TODO: Implementa esto
    # Pistas:
    # - Empieza con zeros
    # - Usa np.eye() o indexación para la diagonal
    # - Usa np.diag() con k=1 y k=-1 para diagonales secundarias
    
    matriz = None
    return matriz


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """Ejecuta verificaciones básicas"""
    print("🧪 Ejecutando tests...\n")
    
    # Test 1: Creación
    arr = np.array([1, 2, 3])
    assert arr.shape == (3,), "Error en shape"
    print("✅ Test 1 pasado")
    
    # Test 2: Zeros
    zeros = np.zeros((2, 3))
    assert zeros.shape == (2, 3), "Error en zeros shape"
    assert zeros.sum() == 0, "zeros debería sumar 0"
    print("✅ Test 2 pasado")
    
    # Test 3: Arange
    arr = np.arange(10)
    assert len(arr) == 10, "arange debería tener 10 elementos"
    print("✅ Test 3 pasado")
    
    print("\n🎉 ¡Tests básicos pasados!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NUMPY - DÍA 1: ARRAYS BÁSICOS")
    print("=" * 60)
    print()
    
    # Descomenta cada función a medida que la completes:
    
    # crear_arrays_basicos()
    # explorar_atributos()
    # crear_arrays_especiales()
    # manipular_formas()
    # operaciones_basicas()
    # agregaciones()
    # comparaciones()
    # comparar_velocidad()
    
    # Desafío
    # matriz = desafio_matriz_especial()
    # print(f"Matriz especial:\n{matriz}")
    
    # Tests
    # run_tests()
    
    print("\n💡 Completa cada función antes de continuar!")
    print("📚 Lee teoria/01_arrays_basicos.md para más información")
