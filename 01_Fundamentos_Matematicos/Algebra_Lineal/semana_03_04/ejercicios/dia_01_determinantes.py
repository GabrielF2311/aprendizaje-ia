"""
Día 1: Ejercicios de Determinantes
Álgebra Lineal Avanzada - Semana 3-4

Temas:
- Cálculo de determinantes 2×2 y 3×3
- Propiedades de determinantes
- Aplicaciones en ML (matriz invertible, volumen)
- Implementación manual y con NumPy
"""

import numpy as np
from typing import List, Union


# ============================================================================
# EJERCICIO 1: Determinante 2×2 (Implementación Manual)
# ============================================================================

def determinante_2x2(matriz: List[List[float]]) -> float:
    """
    Calcula el determinante de una matriz 2×2 usando la fórmula:
    det(A) = a*d - b*c
    
    Args:
        matriz: Matriz 2×2 como lista de listas
        
    Returns:
        Determinante de la matriz
        
    Ejemplo:
        >>> A = [[3, 8], [4, 6]]
        >>> determinante_2x2(A)
        -14.0
    """
    # TODO: Implementar fórmula det = ad - bc
    a, b = matriz[0]
    c, d = matriz[1]
    
    return a * d - b * c


def test_determinante_2x2():
    """Pruebas para determinante 2×2"""
    print("=" * 60)
    print("EJERCICIO 1: Determinante 2×2")
    print("=" * 60)
    
    # Caso 1: Matriz básica
    A = [[3, 8], [4, 6]]
    det_manual = determinante_2x2(A)
    det_numpy = np.linalg.det(A)
    
    print(f"\nMatriz A:")
    print(np.array(A))
    print(f"Determinante (manual): {det_manual}")
    print(f"Determinante (NumPy): {det_numpy:.1f}")
    print(f"✓ Correcto: {np.isclose(det_manual, det_numpy)}")
    
    # Caso 2: Matriz identidad (det = 1)
    I = [[1, 0], [0, 1]]
    det_I = determinante_2x2(I)
    print(f"\nMatriz identidad: det = {det_I} (debería ser 1)")
    
    # Caso 3: Matriz singular (det = 0)
    S = [[2, 4], [1, 2]]
    det_S = determinante_2x2(S)
    print(f"Matriz singular: det = {det_S} (debería ser 0)")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 2: Determinante 3×3 (Regla de Sarrus)
# ============================================================================

def determinante_3x3(matriz: List[List[float]]) -> float:
    """
    Calcula el determinante de una matriz 3×3 usando la regla de Sarrus.
    
    det(A) = a(ei − fh) − b(di − fg) + c(dh − eg)
    
    Args:
        matriz: Matriz 3×3
        
    Returns:
        Determinante
        
    Ejemplo:
        >>> A = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
        >>> determinante_3x3(A)
        -306.0
    """
    a, b, c = matriz[0]
    d, e, f = matriz[1]
    g, h, i = matriz[2]
    
    # Regla de Sarrus
    det = (a * e * i + b * f * g + c * d * h) - \
          (c * e * g + b * d * i + a * f * h)
    
    return det


def test_determinante_3x3():
    """Pruebas para determinante 3×3"""
    print("=" * 60)
    print("EJERCICIO 2: Determinante 3×3 (Regla de Sarrus)")
    print("=" * 60)
    
    # Caso 1
    A = [[6, 1, 1], 
         [4, -2, 5], 
         [2, 8, 7]]
    
    det_manual = determinante_3x3(A)
    det_numpy = np.linalg.det(A)
    
    print(f"\nMatriz A:")
    print(np.array(A))
    print(f"Determinante (manual): {det_manual}")
    print(f"Determinante (NumPy): {det_numpy:.1f}")
    print(f"✓ Correcto: {np.isclose(det_manual, det_numpy)}")
    
    # Caso 2: Matriz identidad 3×3
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    det_I = determinante_3x3(I)
    print(f"\nMatriz identidad 3×3: det = {det_I}")
    
    # Caso 3: Matriz con fila de ceros (det = 0)
    Z = [[1, 2, 3], [0, 0, 0], [4, 5, 6]]
    det_Z = determinante_3x3(Z)
    print(f"Matriz con fila de ceros: det = {det_Z}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 3: Propiedades de Determinantes
# ============================================================================

def verificar_propiedades_determinantes():
    """
    Verifica propiedades importantes de determinantes:
    1. det(A^T) = det(A)
    2. det(AB) = det(A) * det(B)
    3. det(kA) = k^n * det(A)
    4. det(A^-1) = 1/det(A)
    """
    print("=" * 60)
    print("EJERCICIO 3: Propiedades de Determinantes")
    print("=" * 60)
    
    A = np.array([[2, 3], 
                  [1, 4]])
    B = np.array([[5, 6], 
                  [7, 8]])
    
    # Propiedad 1: det(A^T) = det(A)
    print("\n1. det(A^T) = det(A)")
    det_A = np.linalg.det(A)
    det_AT = np.linalg.det(A.T)
    print(f"   det(A) = {det_A:.2f}")
    print(f"   det(A^T) = {det_AT:.2f}")
    print(f"   ✓ Iguales: {np.isclose(det_A, det_AT)}")
    
    # Propiedad 2: det(AB) = det(A) * det(B)
    print("\n2. det(AB) = det(A) × det(B)")
    det_AB = np.linalg.det(A @ B)
    producto_dets = det_A * np.linalg.det(B)
    print(f"   det(AB) = {det_AB:.2f}")
    print(f"   det(A) × det(B) = {producto_dets:.2f}")
    print(f"   ✓ Iguales: {np.isclose(det_AB, producto_dets)}")
    
    # Propiedad 3: det(kA) = k^n * det(A)
    print("\n3. det(kA) = k^n × det(A)")
    k = 3
    n = A.shape[0]  # dimensión
    det_kA = np.linalg.det(k * A)
    esperado = (k ** n) * det_A
    print(f"   det({k}A) = {det_kA:.2f}")
    print(f"   {k}^{n} × det(A) = {esperado:.2f}")
    print(f"   ✓ Iguales: {np.isclose(det_kA, esperado)}")
    
    # Propiedad 4: det(A^-1) = 1/det(A)
    print("\n4. det(A^-1) = 1/det(A)")
    A_inv = np.linalg.inv(A)
    det_A_inv = np.linalg.det(A_inv)
    inverso_det = 1 / det_A
    print(f"   det(A^-1) = {det_A_inv:.4f}")
    print(f"   1/det(A) = {inverso_det:.4f}")
    print(f"   ✓ Iguales: {np.isclose(det_A_inv, inverso_det)}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 4: Determinante y Matriz Invertible
# ============================================================================

def es_invertible(matriz: np.ndarray, tolerancia: float = 1e-10) -> bool:
    """
    Determina si una matriz es invertible verificando su determinante.
    
    Args:
        matriz: Matriz cuadrada
        tolerancia: Tolerancia para considerar det ≈ 0
        
    Returns:
        True si la matriz es invertible, False en caso contrario
    """
    det = np.linalg.det(matriz)
    return abs(det) > tolerancia


def test_invertibilidad():
    """Pruebas de invertibilidad usando determinantes"""
    print("=" * 60)
    print("EJERCICIO 4: Determinante y Matriz Invertible")
    print("=" * 60)
    
    matrices = {
        "Invertible": np.array([[4, 7], [2, 6]]),
        "Singular (det=0)": np.array([[1, 2], [2, 4]]),
        "Identidad": np.eye(3),
        "Filas proporcionales": np.array([[1, 2, 3], [2, 4, 6], [5, 6, 7]])
    }
    
    for nombre, A in matrices.items():
        det = np.linalg.det(A)
        invertible = es_invertible(A)
        
        print(f"\n{nombre}:")
        print(A)
        print(f"det(A) = {det:.6f}")
        print(f"¿Invertible? {invertible}")
        
        if invertible:
            try:
                A_inv = np.linalg.inv(A)
                print(f"✓ Inversa calculada exitosamente")
                # Verificar A @ A^-1 = I
                producto = A @ A_inv
                es_identidad = np.allclose(producto, np.eye(A.shape[0]))
                print(f"✓ A @ A^-1 = I: {es_identidad}")
            except:
                print("✗ Error al calcular inversa")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 5: Volumen de Paralelepípedo
# ============================================================================

def volumen_paralelipipedo(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    Calcula el volumen de un paralelepípedo formado por 3 vectores.
    
    El volumen es el valor absoluto del determinante de la matriz formada
    por los vectores como filas (o columnas).
    
    Args:
        v1, v2, v3: Vectores 3D
        
    Returns:
        Volumen del paralelepípedo
    """
    # Formar matriz con vectores como filas
    A = np.vstack([v1, v2, v3])
    
    # Volumen = |det(A)|
    volumen = abs(np.linalg.det(A))
    
    return volumen


def test_volumen():
    """Pruebas de cálculo de volumen"""
    print("=" * 60)
    print("EJERCICIO 5: Volumen de Paralelepípedo")
    print("=" * 60)
    
    # Caso 1: Vectores base (cubo unitario)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    
    vol = volumen_paralelipipedo(v1, v2, v3)
    print(f"\nCaso 1: Vectores base (cubo unitario)")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v3 = {v3}")
    print(f"Volumen = {vol} (debería ser 1)")
    
    # Caso 2: Cubo de lado 2
    v1 = np.array([2, 0, 0])
    v2 = np.array([0, 2, 0])
    v3 = np.array([0, 0, 2])
    
    vol = volumen_paralelipipedo(v1, v2, v3)
    print(f"\nCaso 2: Cubo de lado 2")
    print(f"Volumen = {vol} (debería ser 8)")
    
    # Caso 3: Vectores coplanares (volumen = 0)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])  # Combinación lineal de v1 y v2
    
    vol = volumen_paralelipipedo(v1, v2, v3)
    print(f"\nCaso 3: Vectores coplanares")
    print(f"Volumen = {vol} (debería ser 0)")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 6: Área de Paralelogramo (2D)
# ============================================================================

def area_paralelogramo(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcula el área de un paralelogramo formado por 2 vectores en 2D.
    
    Args:
        v1, v2: Vectores 2D
        
    Returns:
        Área del paralelogramo
    """
    # Formar matriz 2×2
    A = np.vstack([v1, v2])
    
    # Área = |det(A)|
    area = abs(np.linalg.det(A))
    
    return area


def test_area():
    """Pruebas de cálculo de área"""
    print("=" * 60)
    print("EJERCICIO 6: Área de Paralelogramo (2D)")
    print("=" * 60)
    
    # Caso 1: Vectores base
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    area = area_paralelogramo(v1, v2)
    print(f"\nVectores base: área = {area} (cuadrado unitario)")
    
    # Caso 2: Rectángulo 3×2
    v1 = np.array([3, 0])
    v2 = np.array([0, 2])
    area = area_paralelogramo(v1, v2)
    print(f"Rectángulo 3×2: área = {area}")
    
    # Caso 3: Vectores paralelos (área = 0)
    v1 = np.array([2, 4])
    v2 = np.array([1, 2])  # v2 = v1/2
    area = area_paralelogramo(v1, v2)
    print(f"Vectores paralelos: área = {area}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 7: Aplicación ML - Verificar Independencia Lineal
# ============================================================================

def son_linealmente_independientes(vectores: List[np.ndarray], 
                                   tolerancia: float = 1e-10) -> bool:
    """
    Verifica si un conjunto de vectores es linealmente independiente.
    
    Método: Si det(A) ≠ 0, los vectores son linealmente independientes.
    (Solo funciona si número de vectores = dimensión)
    
    Args:
        vectores: Lista de vectores
        tolerancia: Tolerancia para det ≈ 0
        
    Returns:
        True si son linealmente independientes
    """
    # Formar matriz con vectores como filas
    A = np.vstack(vectores)
    
    # Verificar que sea cuadrada
    if A.shape[0] != A.shape[1]:
        # Para casos no cuadrados, usar rango
        return np.linalg.matrix_rank(A) == len(vectores)
    
    # Calcular determinante
    det = np.linalg.det(A)
    
    return abs(det) > tolerancia


def test_independencia_lineal():
    """Pruebas de independencia lineal"""
    print("=" * 60)
    print("EJERCICIO 7: Independencia Lineal")
    print("=" * 60)
    
    casos = [
        ("Independientes", [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]),
        ("Dependientes (v3 = v1 + v2)", [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 1, 0])
        ]),
        ("Independientes (no ortogonales)", [
            np.array([1, 2]),
            np.array([3, 4])
        ])
    ]
    
    for nombre, vectores in casos:
        A = np.vstack(vectores)
        det = np.linalg.det(A) if A.shape[0] == A.shape[1] else None
        independientes = son_linealmente_independientes(vectores)
        
        print(f"\n{nombre}:")
        print("Matriz:")
        print(A)
        if det is not None:
            print(f"det(A) = {det:.6f}")
        print(f"¿Linealmente independientes? {independientes}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 8: Proyecto - Sistema de Recomendación
# ============================================================================

def matriz_factorizable(A: np.ndarray) -> bool:
    """
    Verifica si una matriz de ratings puede ser factorizada (det ≠ 0).
    
    En sistemas de recomendación, det(A) ≠ 0 indica que la matriz
    tiene información suficiente para factorización.
    
    Args:
        A: Matriz de ratings
        
    Returns:
        True si es factorizable
    """
    if A.shape[0] != A.shape[1]:
        # Para matrices no cuadradas, verificar rango completo
        return np.linalg.matrix_rank(A) == min(A.shape)
    
    return abs(np.linalg.det(A)) > 1e-10


def test_sistema_recomendacion():
    """Simulación de sistema de recomendación"""
    print("=" * 60)
    print("EJERCICIO 8: Sistema de Recomendación")
    print("=" * 60)
    
    # Matriz de ratings (usuarios × películas)
    ratings_completa = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4]
    ])
    
    # Submatriz cuadrada para analizar
    submatriz = ratings_completa[:3, :3]
    
    print("\nMatriz de ratings (usuarios × películas):")
    print(ratings_completa)
    
    print(f"\nSubmatriz 3×3:")
    print(submatriz)
    
    det = np.linalg.det(submatriz)
    print(f"det(submatriz) = {det:.2f}")
    
    factorizable = matriz_factorizable(submatriz)
    print(f"¿Factorizable? {factorizable}")
    
    if factorizable:
        print("✓ La matriz tiene información suficiente para factorización")
    else:
        print("✗ La matriz es singular, puede haber problemas de factorización")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta todos los ejercicios"""
    print("\n" + "=" * 60)
    print("EJERCICIOS DÍA 1: DETERMINANTES")
    print("Álgebra Lineal Avanzada - Semana 3-4")
    print("=" * 60 + "\n")
    
    test_determinante_2x2()
    test_determinante_3x3()
    verificar_propiedades_determinantes()
    test_invertibilidad()
    test_volumen()
    test_area()
    test_independencia_lineal()
    test_sistema_recomendacion()
    
    print("=" * 60)
    print("✓ TODOS LOS EJERCICIOS COMPLETADOS")
    print("=" * 60)


if __name__ == "__main__":
    main()
