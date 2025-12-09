"""
Día 3: Ejercicios de Eigenvalores y Eigenvectores
Álgebra Lineal Avanzada - Semana 3-4

Temas:
- Cálculo de eigenvalores (método característico)
- Cálculo de eigenvectores
- Verificación de la ecuación Av = λv
- Diagonalización de matrices
- Potencias de matrices usando eigendecomposición
- Aplicaciones en ML (cadenas de Markov, PageRank)
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


# ============================================================================
# EJERCICIO 1: Calcular Eigenvalores Manualmente (2×2)
# ============================================================================

def eigenvalores_2x2(A: np.ndarray) -> np.ndarray:
    """
    Calcula eigenvalores de una matriz 2×2 resolviendo:
    det(A - λI) = 0
    
    Para matriz [[a, b], [c, d]]:
    λ² - (a+d)λ + (ad-bc) = 0
    
    Args:
        A: Matriz 2×2
        
    Returns:
        Array con los 2 eigenvalores
    """
    if A.shape != (2, 2):
        raise ValueError("La matriz debe ser 2×2")
    
    a, b = A[0]
    c, d = A[1]
    
    # Coeficientes del polinomio característico: λ² - traza·λ + det
    traza = a + d
    det = a * d - b * c
    
    # Resolver ecuación cuadrática: λ² - traza·λ + det = 0
    # λ = (traza ± √(traza² - 4·det)) / 2
    discriminante = traza**2 - 4*det
    
    if discriminante >= 0:
        lambda1 = (traza + np.sqrt(discriminante)) / 2
        lambda2 = (traza - np.sqrt(discriminante)) / 2
    else:
        # Eigenvalores complejos
        real = traza / 2
        imag = np.sqrt(-discriminante) / 2
        lambda1 = complex(real, imag)
        lambda2 = complex(real, -imag)
    
    return np.array([lambda1, lambda2])


def test_eigenvalores_2x2():
    """Pruebas para eigenvalores 2×2"""
    print("=" * 60)
    print("EJERCICIO 1: Eigenvalores 2×2 (Manual)")
    print("=" * 60)
    
    # Caso 1: Matriz simétrica
    A = np.array([[4, 1],
                  [1, 3]], dtype=float)
    
    print(f"\nMatriz A:")
    print(A)
    
    eigenvalues_manual = eigenvalores_2x2(A)
    eigenvalues_numpy = np.linalg.eigvals(A)
    
    print(f"\nEigenvalores (manual): {eigenvalues_manual}")
    print(f"Eigenvalores (NumPy): {eigenvalues_numpy}")
    print(f"✓ Iguales: {np.allclose(eigenvalues_manual, eigenvalues_numpy)}")
    
    # Caso 2: Matriz identidad (eigenvalores = 1)
    I = np.eye(2)
    eig_I = eigenvalores_2x2(I)
    print(f"\nMatriz identidad: eigenvalores = {eig_I}")
    
    # Caso 3: Matriz de rotación (eigenvalores complejos)
    theta = np.pi / 4  # 45 grados
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    eig_R = eigenvalores_2x2(R)
    print(f"\nMatriz rotación 45°: eigenvalores = {eig_R}")
    print(f"(Complejos con magnitud 1)")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 2: Calcular Eigenvectores
# ============================================================================

def eigenvector(A: np.ndarray, lambda_val: float, tolerancia: float = 1e-10) -> np.ndarray:
    """
    Calcula el eigenvector asociado a un eigenvalor resolviendo:
    (A - λI)v = 0
    
    Args:
        A: Matriz cuadrada
        lambda_val: Eigenvalor
        tolerancia: Tolerancia numérica
        
    Returns:
        Eigenvector normalizado
    """
    n = A.shape[0]
    
    # Construir (A - λI)
    A_lambda = A - lambda_val * np.eye(n)
    
    # El eigenvector es el nullspace de (A - λI)
    # Usamos SVD para encontrar el vector en el nullspace
    U, S, VT = np.linalg.svd(A_lambda)
    
    # El eigenvector es la última columna de V (última fila de VT)
    v = VT[-1]
    
    # Normalizar
    v = v / np.linalg.norm(v)
    
    return v


def test_eigenvectores():
    """Pruebas para eigenvectores"""
    print("=" * 60)
    print("EJERCICIO 2: Calcular Eigenvectores")
    print("=" * 60)
    
    A = np.array([[4, -2],
                  [1, 1]], dtype=float)
    
    print(f"\nMatriz A:")
    print(A)
    
    # Obtener eigenvalores y eigenvectores con NumPy
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"\nEigenvalores: {eigenvalues}")
    
    # Calcular eigenvector manualmente para el primer eigenvalor
    lambda1 = eigenvalues[0]
    v1_manual = eigenvector(A, lambda1)
    v1_numpy = eigenvectors[:, 0]
    
    print(f"\nPara λ₁ = {lambda1:.4f}:")
    print(f"Eigenvector (manual): {v1_manual}")
    print(f"Eigenvector (NumPy): {v1_numpy}")
    
    # Los eigenvectores pueden diferir en signo
    iguales = np.allclose(v1_manual, v1_numpy) or np.allclose(v1_manual, -v1_numpy)
    print(f"✓ Iguales (permitiendo signo opuesto): {iguales}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 3: Verificar Ecuación Av = λv
# ============================================================================

def verificar_ecuacion_eigen(A: np.ndarray, lambda_val: float, v: np.ndarray) -> bool:
    """
    Verifica que Av = λv
    
    Args:
        A: Matriz
        lambda_val: Eigenvalor
        v: Eigenvector
        
    Returns:
        True si cumple la ecuación
    """
    Av = A @ v
    lambda_v = lambda_val * v
    
    return np.allclose(Av, lambda_v)


def test_verificacion_eigen():
    """Pruebas de verificación de ecuación eigen"""
    print("=" * 60)
    print("EJERCICIO 3: Verificar Av = λv")
    print("=" * 60)
    
    A = np.array([[3, 1],
                  [0, 2]], dtype=float)
    
    print(f"\nMatriz A:")
    print(A)
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        v_i = eigenvectors[:, i]
        
        Av = A @ v_i
        lambda_v = lambda_i * v_i
        
        print(f"\nλ_{i+1} = {lambda_i:.4f}")
        print(f"v_{i+1} = {v_i}")
        print(f"Av = {Av}")
        print(f"λv = {lambda_v}")
        
        es_correcto = verificar_ecuacion_eigen(A, lambda_i, v_i)
        print(f"✓ Av = λv: {es_correcto}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 4: Diagonalización A = QΛQ⁻¹
# ============================================================================

def diagonalizar(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonaliza una matriz: A = Q Λ Q⁻¹
    
    Args:
        A: Matriz cuadrada
        
    Returns:
        Q (eigenvectores), Λ (eigenvalores diagonal), Q⁻¹
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    Q = eigenvectors
    Lambda = np.diag(eigenvalues)
    Q_inv = np.linalg.inv(Q)
    
    return Q, Lambda, Q_inv


def test_diagonalizacion():
    """Pruebas de diagonalización"""
    print("=" * 60)
    print("EJERCICIO 4: Diagonalización A = QΛQ⁻¹")
    print("=" * 60)
    
    A = np.array([[4, 1],
                  [2, 3]], dtype=float)
    
    print(f"\nMatriz A:")
    print(A)
    
    Q, Lambda, Q_inv = diagonalizar(A)
    
    print(f"\nQ (eigenvectores):")
    print(Q)
    print(f"\nΛ (eigenvalores diagonal):")
    print(Lambda)
    print(f"\nQ⁻¹:")
    print(Q_inv)
    
    # Reconstruir A
    A_reconstruida = Q @ Lambda @ Q_inv
    
    print(f"\nA reconstruida (QΛQ⁻¹):")
    print(A_reconstruida)
    
    print(f"\n✓ A = QΛQ⁻¹: {np.allclose(A, A_reconstruida)}")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 5: Potencias de Matriz usando Eigendecomposición
# ============================================================================

def potencia_matriz(A: np.ndarray, n: int) -> np.ndarray:
    """
    Calcula A^n eficientemente usando eigendecomposición:
    A^n = Q Λ^n Q⁻¹
    
    Args:
        A: Matriz cuadrada
        n: Exponente
        
    Returns:
        A^n
    """
    Q, Lambda, Q_inv = diagonalizar(A)
    
    # Λ^n es fácil: elevar cada elemento diagonal a n
    Lambda_n = np.diag(np.diag(Lambda) ** n)
    
    # A^n = Q Λ^n Q⁻¹
    A_n = Q @ Lambda_n @ Q_inv
    
    return A_n


def test_potencia_matriz():
    """Pruebas de potencias de matrices"""
    print("=" * 60)
    print("EJERCICIO 5: A^n usando Eigendecomposición")
    print("=" * 60)
    
    A = np.array([[2, 1],
                  [1, 2]], dtype=float)
    
    print(f"\nMatriz A:")
    print(A)
    
    # Calcular A^10
    n = 10
    
    # Método 1: Multiplicación directa (lento)
    import time
    start = time.time()
    A_n_directo = np.linalg.matrix_power(A, n)
    t_directo = time.time() - start
    
    # Método 2: Eigendecomposición (rápido para n grande)
    start = time.time()
    A_n_eigen = potencia_matriz(A, n)
    t_eigen = time.time() - start
    
    print(f"\nA^{n} (método directo):")
    print(A_n_directo)
    print(f"Tiempo: {t_directo:.6f}s")
    
    print(f"\nA^{n} (eigendecomposición):")
    print(A_n_eigen)
    print(f"Tiempo: {t_eigen:.6f}s")
    
    print(f"\n✓ Resultados iguales: {np.allclose(A_n_directo, A_n_eigen)}")
    
    # Para n muy grande, eigendecomposición es mucho más rápido
    n_grande = 100
    A_100_eigen = potencia_matriz(A, n_grande)
    print(f"\nA^{n_grande} calculado con eigendecomposición:")
    print(f"(valores muy grandes, pero cálculo rápido)")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 6: Visualización Geométrica
# ============================================================================

def visualizar_eigenvectores(A: np.ndarray):
    """
    Visualiza eigenvectores y cómo A transforma el espacio
    """
    print("=" * 60)
    print("EJERCICIO 6: Visualización Geométrica")
    print("=" * 60)
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Círculo unitario
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transformar círculo
    ellipse = A @ circle
    
    # Plot 1: Eigenvectores
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Eigenvectores de A')
    
    colors = ['red', 'blue']
    for i in range(2):
        v = eigenvectors[:, i]
        lambda_i = eigenvalues[i]
        
        # Dibujar eigenvector original
        ax1.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2,
                 fc=colors[i], ec=colors[i], linewidth=2, 
                 label=f'v{i+1}', alpha=0.5)
        
        # Dibujar Av = λv
        Av = lambda_i * v
        ax1.arrow(0, 0, Av[0], Av[1], head_width=0.2, head_length=0.2,
                 fc=colors[i], ec=colors[i], linewidth=2, linestyle='--',
                 label=f'λ{i+1}v{i+1}', alpha=0.8)
    
    ax1.legend()
    
    # Plot 2: Transformación
    ax2.plot(circle[0], circle[1], 'b-', label='Círculo original', alpha=0.5)
    ax2.plot(ellipse[0], ellipse[1], 'r-', label='Después de A', linewidth=2)
    
    # Eigenvectores sobre la transformación
    for i in range(2):
        v = eigenvectors[:, i] * 2  # Escalar para visualización
        lambda_i = eigenvalues[i]
        Av = lambda_i * v
        
        ax2.arrow(0, 0, v[0], v[1], head_width=0.3, head_length=0.3,
                 fc=colors[i], ec=colors[i], linewidth=2, alpha=0.3)
        ax2.arrow(0, 0, Av[0], Av[1], head_width=0.3, head_length=0.3,
                 fc=colors[i], ec=colors[i], linewidth=2, linestyle='--')
    
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title('Transformación: círculo → elipse')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('eigenvectores_visualizacion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nMatriz A:")
    print(A)
    print(f"\nEigenvalores: {eigenvalues}")
    print(f"\nEigenvectores:")
    print(eigenvectors)
    print("\n✓ Gráfica guardada como 'eigenvectores_visualizacion.png'")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 7: Cadena de Markov
# ============================================================================

def estado_estacionario_markov(P: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Encuentra el estado estacionario de una cadena de Markov.
    
    El estado estacionario π satisface: πP = π
    Es el eigenvector asociado al eigenvalor λ = 1
    
    Args:
        P: Matriz de transición (cada fila suma 1)
        max_iter: Iteraciones máximas
        
    Returns:
        Vector de estado estacionario
    """
    # El estado estacionario es el eigenvector de P^T con eigenvalor 1
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Encontrar índice del eigenvalor más cercano a 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    
    # Eigenvector correspondiente
    pi = np.real(eigenvectors[:, idx])
    
    # Normalizar para que sume 1 (distribución de probabilidad)
    pi = pi / np.sum(pi)
    
    return pi


def test_cadena_markov():
    """Pruebas de cadena de Markov"""
    print("=" * 60)
    print("EJERCICIO 7: Cadena de Markov - Estado Estacionario")
    print("=" * 60)
    
    # Ejemplo: Clima (Soleado/Lluvioso)
    # P[i,j] = probabilidad de pasar de estado i a estado j
    P = np.array([[0.9, 0.1],   # Soleado → Soleado/Lluvioso
                  [0.5, 0.5]])   # Lluvioso → Soleado/Lluvioso
    
    print("\nCadena de Markov del clima:")
    print("Estados: [Soleado, Lluvioso]")
    print("\nMatriz de transición P:")
    print(P)
    print("\nP[0,1] = 0.1 → Si hoy es soleado, 10% probabilidad de lluvia mañana")
    print("P[1,0] = 0.5 → Si hoy llueve, 50% probabilidad de sol mañana")
    
    # Estado estacionario
    pi = estado_estacionario_markov(P)
    
    print(f"\nEstado estacionario π: {pi}")
    print(f"A largo plazo:")
    print(f"  {pi[0]*100:.1f}% de días soleados")
    print(f"  {pi[1]*100:.1f}% de días lluviosos")
    
    # Verificar que πP = π
    pi_P = pi @ P
    print(f"\nVerificación πP = π:")
    print(f"πP = {pi_P}")
    print(f"π  = {pi}")
    print(f"✓ Correcto: {np.allclose(pi_P, pi)}")
    
    # Simular evolución
    print("\nEvolución desde estado inicial [1, 0] (100% soleado):")
    estado = np.array([1.0, 0.0])
    for dia in [1, 2, 5, 10, 50]:
        estado_dia = estado @ np.linalg.matrix_power(P, dia)
        print(f"Día {dia:2d}: {estado_dia} (Sol: {estado_dia[0]*100:.1f}%)")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# EJERCICIO 8: PageRank Simplificado
# ============================================================================

def pagerank_simple(enlaces: np.ndarray, d: float = 0.85, 
                    max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Algoritmo PageRank simplificado.
    
    Args:
        enlaces: Matriz de adyacencia (enlaces[i,j] = 1 si i enlaza a j)
        d: Factor de amortiguamiento (damping factor)
        max_iter: Iteraciones máximas
        tol: Tolerancia de convergencia
        
    Returns:
        Scores de PageRank normalizados
    """
    n = enlaces.shape[0]
    
    # Normalizar matriz de enlaces (cada fila suma 1)
    out_degree = enlaces.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1  # Evitar división por 0
    M = enlaces / out_degree
    
    # Matriz de Google: G = dM + (1-d)/n * ee^T
    G = d * M + (1 - d) / n * np.ones((n, n))
    
    # Power iteration para encontrar eigenvector dominante
    pagerank = np.ones(n) / n  # Inicializar uniforme
    
    for _ in range(max_iter):
        pagerank_new = pagerank @ G
        
        if np.linalg.norm(pagerank_new - pagerank) < tol:
            break
        
        pagerank = pagerank_new
    
    # Normalizar
    pagerank = pagerank / np.sum(pagerank)
    
    return pagerank


def test_pagerank():
    """Pruebas de PageRank"""
    print("=" * 60)
    print("EJERCICIO 8: PageRank Simplificado")
    print("=" * 60)
    
    # Red de páginas web simple
    # Páginas: A, B, C, D
    enlaces = np.array([
        [0, 1, 1, 0],  # A enlaza a B y C
        [1, 0, 0, 0],  # B enlaza a A
        [0, 1, 0, 1],  # C enlaza a B y D
        [0, 1, 1, 0]   # D enlaza a B y C
    ], dtype=float)
    
    print("\nRed de páginas web:")
    print("A → B, C")
    print("B → A")
    print("C → B, D")
    print("D → B, C")
    
    print("\nMatriz de enlaces:")
    print(enlaces)
    
    # Calcular PageRank
    pr = pagerank_simple(enlaces)
    
    paginas = ['A', 'B', 'C', 'D']
    print("\nPageRank scores:")
    for i, pagina in enumerate(paginas):
        print(f"  Página {pagina}: {pr[i]:.4f}")
    
    # Página más importante
    idx_max = np.argmax(pr)
    print(f"\n✓ Página más importante: {paginas[idx_max]} (score: {pr[idx_max]:.4f})")
    print("  (Recibe más enlaces entrantes)")
    
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta todos los ejercicios"""
    print("\n" + "=" * 60)
    print("EJERCICIOS DÍA 3: EIGENVALORES Y EIGENVECTORES")
    print("Álgebra Lineal Avanzada - Semana 3-4")
    print("=" * 60 + "\n")
    
    test_eigenvalores_2x2()
    test_eigenvectores()
    test_verificacion_eigen()
    test_diagonalizacion()
    test_potencia_matriz()
    
    # Visualización (comentar si no tienes matplotlib)
    A_viz = np.array([[2, 1], [1, 2]], dtype=float)
    visualizar_eigenvectores(A_viz)
    
    test_cadena_markov()
    test_pagerank()
    
    print("=" * 60)
    print("✓ TODOS LOS EJERCICIOS COMPLETADOS")
    print("=" * 60)


if __name__ == "__main__":
    main()
