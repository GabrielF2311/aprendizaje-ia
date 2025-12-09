"""
Día 2: Ejercicios de Matriz Inversa
Álgebra Lineal Avanzada - Semana 3-4

Temas:
- Cálculo de matriz inversa (2×2, 3×3)
- Verificación de inversas
- Resolución de sistemas usando inversas
- Pseudoinversa para matrices no cuadradas
- Aplicaciones en ML
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================================
# EJERCICIO 1: Inversa de Matriz 2×2 (Fórmula Directa)
# ============================================================================

def inversa_2x2(A: np.ndarray) -> Optional[np.ndarray]:
    """
    Calcula la inversa de una matriz 2×2 usando la fórmula:
    
    A^-1 = (1/det(A)) * [[d, -b], [-c, a]]
    
    Args:
        A: Matriz 2×2
        
    Returns:
        Matriz inversa o None si no es invertible
    """
    if A.shape != (2, 2):
        raise ValueError("La matriz debe ser 2×2")
    
    # Elementos de la matriz
    a, b = A[0]
    c, d = A[1]
    
    # Calcular determinante
    det = a * d - b * c
    
    # Verificar si es invertible
    if abs(det) < 1e-10:
        print("⚠️ Matriz singular (det ≈ 0), no es invertible")
        return None
    
    # Fórmula de inversa 2×2
    A_inv = (1 / det) * np.array([[d, -b],
                                   [-c, a]])
    
    return A_inv


def test_inversa_2x2():
    """Pruebas para inversa 2×2"""
    print("=" * 60)
    print("EJERCICIO 1: Inversa de Matriz 2×2")
    print("=" * 60)
    
    # Caso 1: Matriz invertible
    A = np.array([[4, 7],
                  [2, 6]], dtype=float)
    
    print(f"\nMatriz A:")
    print(A)
    
    A_inv_manual = inversa_2x2(A)
    A_inv_numpy = np.linalg.inv(A)
    
    print(f"\nInversa (manual):")
    print(A_inv_manual)
    print(f"\nInversa (NumPy):")
    print(A_inv_numpy)
    
    # Verificar A @ A^-1 = I
    producto = A @ A_inv_manual
    print(f"\nA @ A^-1:")
    print(producto)
    print(f"✓ Es identidad: {np.allclose(producto, np.eye(2))}")
    
    # Caso 2: Matriz singular
    print("\n" + "-" * 60)
    S = np.array([[2, 4],
                  [1, 2]], dtype=float)
    
    print(f"\nMatriz singular S:")
    print(S)
    print(f"det(S) = {np.linalg.det(S):.6f}")
    
    S_inv = inversa_2x2(S)
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 2: Inversa usando Eliminación de Gauss-Jordan
# ============================================================================

def inversa_gauss_jordan(A: np.ndarray) -> Optional[np.ndarray]:
    """
    Calcula la inversa usando eliminación de Gauss-Jordan.
    
    Método: [A | I] → [I | A^-1]
    
    Args:
        A: Matriz cuadrada
        
    Returns:
        Matriz inversa o None si no es invertible
    """
    n = A.shape[0]
    
    # Verificar que sea cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada")
    
    # Crear matriz aumentada [A | I]
    augmented = np.hstack([A.astype(float), np.eye(n)])
    
    # Eliminación hacia adelante y atrás
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # Verificar pivote no nulo
        if abs(augmented[i, i]) < 1e-10:
            print("⚠️ Matriz singular, no invertible")
            return None
        
        # Normalizar fila del pivote
        augmented[i] = augmented[i] / augmented[i, i]
        
        # Eliminar en otras filas
        for j in range(n):
            if i != j:
                augmented[j] -= augmented[j, i] * augmented[i]
    
    # Extraer la inversa (lado derecho)
    A_inv = augmented[:, n:]
    
    return A_inv


def test_gauss_jordan():
    """Pruebas de Gauss-Jordan"""
    print("=" * 60)
    print("EJERCICIO 2: Inversa por Gauss-Jordan")
    print("=" * 60)
    
    A = np.array([[2, 1, 0],
                  [1, 2, 1],
                  [0, 1, 2]], dtype=float)
    
    print(f"\nMatriz A (3×3):")
    print(A)
    
    A_inv_gj = inversa_gauss_jordan(A)
    A_inv_numpy = np.linalg.inv(A)
    
    print(f"\nInversa (Gauss-Jordan):")
    print(A_inv_gj)
    print(f"\nInversa (NumPy):")
    print(A_inv_numpy)
    
    print(f"\n✓ Iguales: {np.allclose(A_inv_gj, A_inv_numpy)}")
    
    # Verificación
    producto = A @ A_inv_gj
    print(f"\nA @ A^-1:")
    print(producto)
    print(f"✓ Es identidad: {np.allclose(producto, np.eye(3))}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 3: Propiedades de Matrices Inversas
# ============================================================================

def verificar_propiedades_inversas():
    """
    Verifica propiedades de matrices inversas:
    1. (A^-1)^-1 = A
    2. (AB)^-1 = B^-1 A^-1
    3. (A^T)^-1 = (A^-1)^T
    4. det(A^-1) = 1/det(A)
    """
    print("=" * 60)
    print("EJERCICIO 3: Propiedades de Matrices Inversas")
    print("=" * 60)
    
    A = np.array([[2, 3],
                  [1, 4]], dtype=float)
    B = np.array([[5, 6],
                  [7, 8]], dtype=float)
    
    A_inv = np.linalg.inv(A)
    B_inv = np.linalg.inv(B)
    
    # Propiedad 1: (A^-1)^-1 = A
    print("\n1. (A^-1)^-1 = A")
    A_inv_inv = np.linalg.inv(A_inv)
    print(f"   (A^-1)^-1:")
    print(f"   {A_inv_inv}")
    print(f"   A:")
    print(f"   {A}")
    print(f"   ✓ Iguales: {np.allclose(A_inv_inv, A)}")
    
    # Propiedad 2: (AB)^-1 = B^-1 A^-1
    print("\n2. (AB)^-1 = B^-1 A^-1")
    AB = A @ B
    AB_inv_directo = np.linalg.inv(AB)
    AB_inv_propiedad = B_inv @ A_inv
    print(f"   ✓ Iguales: {np.allclose(AB_inv_directo, AB_inv_propiedad)}")
    
    # Propiedad 3: (A^T)^-1 = (A^-1)^T
    print("\n3. (A^T)^-1 = (A^-1)^T")
    AT_inv = np.linalg.inv(A.T)
    A_inv_T = A_inv.T
    print(f"   ✓ Iguales: {np.allclose(AT_inv, A_inv_T)}")
    
    # Propiedad 4: det(A^-1) = 1/det(A)
    print("\n4. det(A^-1) = 1/det(A)")
    det_A = np.linalg.det(A)
    det_A_inv = np.linalg.det(A_inv)
    inverso_det = 1 / det_A
    print(f"   det(A^-1) = {det_A_inv:.6f}")
    print(f"   1/det(A) = {inverso_det:.6f}")
    print(f"   ✓ Iguales: {np.isclose(det_A_inv, inverso_det)}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 4: Resolver Sistema Ax=b usando Inversa
# ============================================================================

def resolver_con_inversa(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Resuelve el sistema Ax = b usando x = A^-1 b
    
    Args:
        A: Matriz de coeficientes (n×n)
        b: Vector de términos independientes (n×1)
        
    Returns:
        Solución x o None si A no es invertible
    """
    try:
        A_inv = np.linalg.inv(A)
        x = A_inv @ b
        return x
    except np.linalg.LinAlgError:
        print("⚠️ Matriz singular, no se puede resolver")
        return None


def test_resolver_sistemas():
    """Pruebas de resolución de sistemas"""
    print("=" * 60)
    print("EJERCICIO 4: Resolver Sistemas con Inversa")
    print("=" * 60)
    
    # Sistema: 2x + 3y = 8
    #          x + 4y = 9
    A = np.array([[2, 3],
                  [1, 4]], dtype=float)
    b = np.array([8, 9], dtype=float)
    
    print("\nSistema Ax = b:")
    print("Matriz A:")
    print(A)
    print(f"Vector b: {b}")
    
    # Método 1: Usando inversa
    x_inversa = resolver_con_inversa(A, b)
    
    # Método 2: np.linalg.solve (más eficiente)
    x_solve = np.linalg.solve(A, b)
    
    print(f"\nSolución (inversa): x = {x_inversa}")
    print(f"Solución (solve): x = {x_solve}")
    print(f"✓ Iguales: {np.allclose(x_inversa, x_solve)}")
    
    # Verificación
    resultado = A @ x_inversa
    print(f"\nVerificación A @ x = {resultado}")
    print(f"b = {b}")
    print(f"✓ Correcto: {np.allclose(resultado, b)}")
    
    # Nota sobre eficiencia
    print("\n" + "-" * 60)
    print("⚠️ NOTA: En ML, usar np.linalg.solve() es más eficiente")
    print("   que calcular la inversa explícitamente.")
    print("   Complejidad: O(n³) para ambos, pero solve es ~2× más rápido")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 5: Pseudoinversa (Moore-Penrose)
# ============================================================================

def pseudoinversa_manual(A: np.ndarray) -> np.ndarray:
    """
    Calcula la pseudoinversa usando la fórmula:
    
    Para matrices sobredeterminadas (m > n):
    A^+ = (A^T A)^-1 A^T
    
    Args:
        A: Matriz m×n
        
    Returns:
        Pseudoinversa A^+
    """
    # Calcular A^T A
    ATA = A.T @ A
    
    # Verificar que sea invertible
    if abs(np.linalg.det(ATA)) < 1e-10:
        print("⚠️ A^T A es singular, usando np.linalg.pinv")
        return np.linalg.pinv(A)
    
    # A^+ = (A^T A)^-1 A^T
    ATA_inv = np.linalg.inv(ATA)
    A_pinv = ATA_inv @ A.T
    
    return A_pinv


def test_pseudoinversa():
    """Pruebas de pseudoinversa"""
    print("=" * 60)
    print("EJERCICIO 5: Pseudoinversa (Moore-Penrose)")
    print("=" * 60)
    
    # Matriz sobredeterminada (más filas que columnas)
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=float)
    
    print(f"\nMatriz A (3×2):")
    print(A)
    
    # Pseudoinversa manual
    A_pinv_manual = pseudoinversa_manual(A)
    
    # Pseudoinversa NumPy
    A_pinv_numpy = np.linalg.pinv(A)
    
    print(f"\nPseudoinversa (manual):")
    print(A_pinv_manual)
    print(f"\nPseudoinversa (NumPy):")
    print(A_pinv_numpy)
    
    print(f"\n✓ Iguales: {np.allclose(A_pinv_manual, A_pinv_numpy)}")
    
    # Verificar propiedad: A @ A^+ @ A = A
    verificacion = A @ A_pinv_manual @ A
    print(f"\nVerificación A @ A^+ @ A = A:")
    print(f"✓ Correcto: {np.allclose(verificacion, A)}")
    
    # Aplicación: Resolver sistema sobredeterminado (mínimos cuadrados)
    b = np.array([1, 2, 3], dtype=float)
    x = A_pinv_manual @ b
    
    print(f"\nResolver Ax ≈ b (mínimos cuadrados):")
    print(f"Solución x = {x}")
    
    # Error
    error = np.linalg.norm(A @ x - b)
    print(f"Error ||Ax - b|| = {error:.6f}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 6: Número de Condición
# ============================================================================

def numero_condicion(A: np.ndarray) -> float:
    """
    Calcula el número de condición de una matriz:
    cond(A) = ||A|| * ||A^-1||
    
    Un número de condición alto indica que la matriz está mal condicionada
    (pequeños cambios en los datos pueden causar grandes cambios en la solución).
    
    Args:
        A: Matriz cuadrada
        
    Returns:
        Número de condición
    """
    return np.linalg.cond(A)


def test_condicionamiento():
    """Pruebas de condicionamiento de matrices"""
    print("=" * 60)
    print("EJERCICIO 6: Número de Condición")
    print("=" * 60)
    
    matrices = {
        "Bien condicionada (Identidad)": np.eye(3),
        "Bien condicionada": np.array([[4, 1], [1, 3]], dtype=float),
        "Mal condicionada": np.array([[1, 1], [1, 1.0001]], dtype=float),
        "Muy mal condicionada": np.array([[1, 1], [1, 1.000001]], dtype=float)
    }
    
    for nombre, A in matrices.items():
        cond = numero_condicion(A)
        
        print(f"\n{nombre}:")
        print(A)
        print(f"cond(A) = {cond:.2e}")
        
        if cond < 10:
            print("✓ Matriz bien condicionada")
        elif cond < 1000:
            print("⚠️ Matriz moderadamente condicionada")
        elif cond < 1e6:
            print("⚠️ Matriz mal condicionada")
        else:
            print("❌ Matriz muy mal condicionada - resultados inestables")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 7: Aplicación ML - Regresión Lineal con Ecuaciones Normales
# ============================================================================

def regresion_lineal_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Resuelve regresión lineal usando ecuaciones normales:
    w = (X^T X)^-1 X^T y
    
    Args:
        X: Matriz de features (n_samples, n_features)
        y: Vector target (n_samples,)
        
    Returns:
        Pesos w (n_features,)
    """
    # Ecuaciones normales: w = (X^T X)^-1 X^T y
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Verificar condicionamiento
    cond = np.linalg.cond(XTX)
    if cond > 1e10:
        print(f"⚠️ Matriz mal condicionada (cond={cond:.2e})")
        print("   Considerar usar regularización o SVD")
    
    # Resolver usando inversa
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv @ XTy
    
    return w


def test_regresion_lineal():
    """Pruebas de regresión lineal"""
    print("=" * 60)
    print("EJERCICIO 7: Regresión Lineal (Ecuaciones Normales)")
    print("=" * 60)
    
    # Datos de ejemplo: y = 2x + 3 + ruido
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1)
    y_true = 2 * X.ravel() + 3
    y = y_true + 0.5 * np.random.randn(n_samples)
    
    # Agregar columna de unos (bias)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    
    print(f"\nDatos: {n_samples} muestras, 1 feature")
    print(f"Modelo verdadero: y = 2x + 3")
    
    # Resolver con ecuaciones normales
    w = regresion_lineal_normal_equations(X_b, y)
    
    print(f"\nPesos estimados:")
    print(f"  Intercepto (bias): {w[0]:.4f} (esperado: 3)")
    print(f"  Pendiente: {w[1]:.4f} (esperado: 2)")
    
    # Comparar con sklearn
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    print(f"\nComparación con sklearn:")
    print(f"  Intercepto: {lr.intercept_:.4f}")
    print(f"  Pendiente: {lr.coef_[0]:.4f}")
    print(f"  ✓ Iguales: {np.allclose([lr.intercept_, lr.coef_[0]], w)}")
    
    # Error MSE
    y_pred = X_b @ w
    mse = np.mean((y - y_pred) ** 2)
    print(f"\nMean Squared Error: {mse:.4f}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 8: Comparación de Métodos
# ============================================================================

def comparar_metodos_resolucion():
    """
    Compara diferentes métodos para resolver Ax = b:
    1. Inversa explícita: x = A^-1 b
    2. np.linalg.solve()
    3. Pseudoinversa: x = A^+ b
    """
    import time
    
    print("=" * 60)
    print("EJERCICIO 8: Comparación de Métodos")
    print("=" * 60)
    
    # Sistema grande
    n = 500
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    
    print(f"\nSistema: {n}×{n}")
    
    # Método 1: Inversa explícita
    start = time.time()
    A_inv = np.linalg.inv(A)
    x1 = A_inv @ b
    t1 = time.time() - start
    
    # Método 2: solve
    start = time.time()
    x2 = np.linalg.solve(A, b)
    t2 = time.time() - start
    
    # Método 3: Pseudoinversa
    start = time.time()
    A_pinv = np.linalg.pinv(A)
    x3 = A_pinv @ b
    t3 = time.time() - start
    
    print(f"\nTiempos de ejecución:")
    print(f"  1. Inversa explícita: {t1:.4f}s")
    print(f"  2. np.linalg.solve(): {t2:.4f}s (más rápido)")
    print(f"  3. Pseudoinversa: {t3:.4f}s")
    
    print(f"\nSpeedup solve vs inversa: {t1/t2:.2f}×")
    
    # Verificar que dan el mismo resultado
    print(f"\n✓ Resultados iguales: {np.allclose(x1, x2) and np.allclose(x2, x3)}")
    
    print("\n💡 CONCLUSIÓN: Usar np.linalg.solve() en lugar de calcular")
    print("   la inversa explícitamente. Es más rápido y numéricamente estable.")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta todos los ejercicios"""
    print("\n" + "=" * 60)
    print("EJERCICIOS DÍA 2: MATRIZ INVERSA")
    print("Álgebra Lineal Avanzada - Semana 3-4")
    print("=" * 60 + "\n")
    
    test_inversa_2x2()
    test_gauss_jordan()
    verificar_propiedades_inversas()
    test_resolver_sistemas()
    test_pseudoinversa()
    test_condicionamiento()
    test_regresion_lineal()
    comparar_metodos_resolucion()
    
    print("=" * 60)
    print("✓ TODOS LOS EJERCICIOS COMPLETADOS")
    print("=" * 60)


if __name__ == "__main__":
    main()
