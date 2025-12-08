"""
PROBABILIDAD - EJERCICIOS PRÁCTICOS
===================================

Ejercicios para dominar conceptos de probabilidad.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# EJERCICIO 1: Probabilidad Básica
# ============================================================================

def probabilidad_basica():
    """
    Ejercicios de probabilidad fundamental.
    """
    print("=" * 60)
    print("EJERCICIO 1: Probabilidad Básica")
    print("=" * 60)
    
    # Experimento: Lanzar un dado
    # TODO: ¿Cuál es la probabilidad de sacar un 6?
    p_seis = None
    
    # TODO: ¿Probabilidad de sacar un número par?
    p_par = None
    
    # TODO: Simula 10000 lanzamientos y verifica
    lanzamientos = None  # np.random.randint(1, 7, size=10000)
    freq_seis = None     # ¿Cuántas veces salió 6?
    prob_empirica = None # freq_seis / 10000
    
    print(f"Probabilidad teórica de 6: {p_seis}")
    print(f"Probabilidad empírica de 6: {prob_empirica:.4f}")


# ============================================================================
# EJERCICIO 2: Teorema de Bayes
# ============================================================================

def ejercicio_bayes():
    """
    Aplicación del Teorema de Bayes.
    
    Problema: Test médico
    - 1% de la población tiene la enfermedad
    - Test tiene 99% de precisión (positivo si enfermo)
    - Test tiene 5% de falsos positivos (positivo si sano)
    
    Si el test da positivo, ¿cuál es la probabilidad de estar enfermo?
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 2: Teorema de Bayes - Test Médico")
    print("=" * 60)
    
    # TODO: Define las probabilidades
    P_enfermo = None          # P(E) = 0.01
    P_positivo_si_enfermo = None  # P(+|E) = 0.99
    P_positivo_si_sano = None     # P(+|S) = 0.05
    
    # TODO: Calcula P(+) usando ley de probabilidad total
    # P(+) = P(+|E)·P(E) + P(+|S)·P(S)
    P_positivo = None
    
    # TODO: Aplica Bayes: P(E|+) = P(+|E)·P(E) / P(+)
    P_enfermo_si_positivo = None
    
    print(f"P(Enfermo | Test Positivo) = {P_enfermo_si_positivo:.4f}")
    print("¡Sorpresa! Incluso con test positivo, es más probable estar sano")


# ============================================================================
# EJERCICIO 3: Distribución Binomial
# ============================================================================

def distribucion_binomial():
    """
    Experimentos binomiales.
    
    Ejemplo: Lanzar 10 monedas, ¿cuál es la probabilidad de 
    obtener exactamente 7 caras?
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 3: Distribución Binomial")
    print("=" * 60)
    
    n = 10      # Número de lanzamientos
    p = 0.5     # Probabilidad de cara
    k = 7       # Número de éxitos deseados
    
    # TODO: Calcula probabilidad usando scipy.stats
    prob = None  # stats.binom.pmf(k, n, p)
    
    print(f"P(7 caras en 10 lanzamientos) = {prob:.4f}")
    
    # TODO: Simula 10000 experimentos de 10 lanzamientos
    experimentos = None  # Genera matriz 10000 x 10
    caras_por_experimento = None  # Suma por fila
    
    # Visualiza
    # plt.hist(caras_por_experimento, bins=11, density=True, alpha=0.7)
    # plt.title('Distribución de caras en 10 lanzamientos')
    # plt.show()


# ============================================================================
# EJERCICIO 4: Distribución Normal
# ============================================================================

def distribucion_normal():
    """
    Trabajar con la distribución normal.
    
    Alturas de personas: μ=170cm, σ=10cm
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 4: Distribución Normal")
    print("=" * 60)
    
    mu = 170    # Media
    sigma = 10  # Desviación estándar
    
    # TODO: ¿Probabilidad de altura > 185cm?
    prob_mayor_185 = None  # 1 - stats.norm.cdf(185, mu, sigma)
    
    # TODO: ¿Probabilidad de altura entre 160 y 180?
    prob_entre = None
    
    # TODO: ¿Qué altura tienen el 95% más bajo de la población?
    altura_percentil_95 = None  # stats.norm.ppf(0.95, mu, sigma)
    
    print(f"P(altura > 185cm) = {prob_mayor_185:.4f}")
    print(f"P(160 < altura < 180) = {prob_entre:.4f}")
    print(f"Percentil 95: {altura_percentil_95:.2f}cm")
    
    # Visualización
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Distribución Normal')
    plt.fill_between(x, y, where=(x >= 160) & (x <= 180), alpha=0.3, 
                     label='160-180cm')
    plt.axvline(mu, color='r', linestyle='--', label=f'Media: {mu}cm')
    plt.xlabel('Altura (cm)')
    plt.ylabel('Densidad')
    plt.title('Distribución de Alturas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# EJERCICIO 5: Esperanza y Varianza
# ============================================================================

def esperanza_varianza():
    """
    Calcular esperanza y varianza de variables aleatorias.
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 5: Esperanza y Varianza")
    print("=" * 60)
    
    # Variable aleatoria discreta
    # X = {1, 2, 3, 4, 5, 6} (dado)
    # P(X=x) = 1/6 para todo x
    
    valores = np.array([1, 2, 3, 4, 5, 6])
    probabilidades = np.array([1/6] * 6)
    
    # TODO: Calcula esperanza E[X] = Σ x·P(x)
    esperanza = None
    
    # TODO: Calcula varianza Var(X) = E[X²] - (E[X])²
    esperanza_cuadrados = None  # Σ x²·P(x)
    varianza = None
    desviacion_std = None
    
    print(f"E[X] = {esperanza}")
    print(f"Var(X) = {varianza:.4f}")
    print(f"σ = {desviacion_std:.4f}")


# ============================================================================
# PROYECTO: SIMULACIÓN MONTE CARLO
# ============================================================================

def estimar_pi():
    """
    Estima π usando método Monte Carlo.
    
    Idea: Lanzar puntos aleatorios en cuadrado [-1,1] x [-1,1]
    Contar cuántos caen dentro del círculo unitario
    π ≈ 4 * (puntos_en_circulo / total_puntos)
    """
    print("\n" + "=" * 60)
    print("PROYECTO: Estimar π con Monte Carlo")
    print("=" * 60)
    
    n = 100000  # Número de puntos
    
    # TODO: Genera puntos aleatorios
    x = None  # np.random.uniform(-1, 1, n)
    y = None  # np.random.uniform(-1, 1, n)
    
    # TODO: Determina cuáles están dentro del círculo
    # Círculo unitario: x² + y² ≤ 1
    dentro_circulo = None
    
    # TODO: Estima π
    pi_estimado = None
    error = None  # abs(pi_estimado - np.pi)
    
    print(f"π estimado: {pi_estimado:.6f}")
    print(f"π real:     {np.pi:.6f}")
    print(f"Error:      {error:.6f}")
    
    # Visualización
    # dentro = x**2 + y**2 <= 1
    # plt.scatter(x[dentro], y[dentro], c='blue', s=1, alpha=0.5)
    # plt.scatter(x[~dentro], y[~dentro], c='red', s=1, alpha=0.5)
    # plt.gca().set_aspect('equal')
    # plt.title(f'Monte Carlo: π ≈ {pi_estimado:.4f}')
    # plt.show()


def problema_monty_hall():
    """
    Simula el famoso problema de Monty Hall.
    
    Demostrar que cambiar de puerta da 2/3 de probabilidad de ganar.
    """
    print("\n" + "=" * 60)
    print("PROYECTO: Problema de Monty Hall")
    print("=" * 60)
    
    n_simulaciones = 10000
    
    # Estrategia 1: No cambiar
    victorias_sin_cambiar = 0
    
    # Estrategia 2: Cambiar siempre
    victorias_cambiando = 0
    
    for _ in range(n_simulaciones):
        # TODO: Simula el juego
        # 1. Premio está en una puerta aleatoria (0, 1, 2)
        puerta_premio = None
        
        # 2. Jugador elige una puerta aleatoria
        eleccion_inicial = None
        
        # 3. Monty abre una puerta (no premio, no elegida)
        # (implementación simplificada)
        
        # 4. Contar victorias
        if eleccion_inicial == puerta_premio:
            victorias_sin_cambiar += 1
        else:
            victorias_cambiando += 1
    
    prob_sin_cambiar = victorias_sin_cambiar / n_simulaciones
    prob_cambiando = victorias_cambiando / n_simulaciones
    
    print(f"P(ganar sin cambiar) = {prob_sin_cambiar:.4f} (teórico: 0.333)")
    print(f"P(ganar cambiando) = {prob_cambiando:.4f} (teórico: 0.667)")


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """Verifica comprensión básica"""
    print("\n🧪 Tests de probabilidad...\n")
    
    # Test 1: Distribución Binomial
    prob = stats.binom.pmf(5, 10, 0.5)
    assert abs(prob - 0.2461) < 0.001, "Error en Binomial"
    print("✅ Test 1: Binomial correcto")
    
    # Test 2: Normal
    prob = stats.norm.cdf(1, 0, 1)
    assert abs(prob - 0.8413) < 0.001, "Error en Normal"
    print("✅ Test 2: Normal correcto")
    
    print("\n🎉 ¡Tests pasados!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROBABILIDAD - EJERCICIOS PRÁCTICOS")
    print("=" * 60)
    
    # Descomenta para ejecutar cada ejercicio:
    
    # probabilidad_basica()
    # ejercicio_bayes()
    # distribucion_binomial()
    # distribucion_normal()
    # esperanza_varianza()
    
    # Proyectos
    # estimar_pi()
    # problema_monty_hall()
    
    # Tests
    # run_tests()
    
    print("\n📚 Lee la teoría en ../README.md")
    print("💡 Completa cada función antes de continuar!")
