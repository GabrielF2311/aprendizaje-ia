"""
ESTADÍSTICA - EJERCICIOS PRÁCTICOS
==================================

Ejercicios para análisis estadístico y machine learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================================
# EJERCICIO 1: Estadísticas Descriptivas
# ============================================================================

def estadisticas_descriptivas():
    """
    Analizar un conjunto de datos con estadísticas básicas.
    """
    print("=" * 60)
    print("EJERCICIO 1: Estadísticas Descriptivas")
    print("=" * 60)
    
    # Dataset: Calificaciones de estudiantes
    calificaciones = np.array([
        85, 92, 78, 90, 88, 76, 95, 89, 84, 91,
        79, 87, 93, 82, 86, 88, 90, 85, 83, 94
    ])
    
    # TODO: Calcula medidas de tendencia central
    media = None
    mediana = None
    moda = None  # stats.mode(calificaciones)
    
    # TODO: Calcula medidas de dispersión
    varianza = None
    desviacion = None
    rango = None  # max - min
    
    # TODO: Calcula cuartiles y percentiles
    q1 = None  # np.percentile(calificaciones, 25)
    q2 = None  # percentil 50
    q3 = None  # percentil 75
    iqr = None  # Rango intercuartílico: Q3 - Q1
    
    print(f"Media: {media:.2f}")
    print(f"Mediana: {mediana:.2f}")
    print(f"Desviación estándar: {desviacion:.2f}")
    print(f"IQR: {iqr:.2f}")
    
    # Visualización: Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(calificaciones, vert=False)
    plt.xlabel('Calificación')
    plt.title('Distribución de Calificaciones')
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# EJERCICIO 2: Correlación
# ============================================================================

def analisis_correlacion():
    """
    Analiza correlación entre variables.
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 2: Análisis de Correlación")
    print("=" * 60)
    
    # Datos: Horas de estudio vs Calificación
    horas_estudio = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    calificacion = np.array([50, 55, 60, 65, 75, 80, 85, 90, 92, 95])
    
    # TODO: Calcula coeficiente de correlación de Pearson
    correlacion = None  # np.corrcoef(horas_estudio, calificacion)[0,1]
    
    # TODO: Calcula covarianza
    covarianza = None  # np.cov(horas_estudio, calificacion)[0,1]
    
    print(f"Correlación (Pearson): {correlacion:.4f}")
    print(f"Covarianza: {covarianza:.4f}")
    
    if correlacion > 0.8:
        print("📈 Correlación positiva fuerte")
    
    # Visualización: Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(horas_estudio, calificacion, s=100, alpha=0.6)
    
    # Línea de tendencia
    z = np.polyfit(horas_estudio, calificacion, 1)
    p = np.poly1d(z)
    plt.plot(horas_estudio, p(horas_estudio), "r--", alpha=0.8, 
             label=f'Tendencia (r={correlacion:.3f})')
    
    plt.xlabel('Horas de Estudio')
    plt.ylabel('Calificación')
    plt.title('Relación Estudio-Calificación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# EJERCICIO 3: Prueba de Hipótesis (t-test)
# ============================================================================

def prueba_t_student():
    """
    Prueba t de Student para comparar dos grupos.
    
    Pregunta: ¿Hay diferencia significativa entre calificaciones
    de dos grupos de estudiantes?
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 3: Prueba t de Student")
    print("=" * 60)
    
    # Datos
    grupo_A = np.array([85, 87, 88, 86, 90, 84, 89, 87, 88, 86])
    grupo_B = np.array([78, 80, 82, 79, 81, 77, 83, 80, 79, 81])
    
    # TODO: Calcula estadísticas básicas
    media_A = None
    media_B = None
    
    # TODO: Realiza prueba t
    # H0: No hay diferencia entre grupos (μA = μB)
    # H1: Hay diferencia (μA ≠ μB)
    
    t_statistic, p_value = None, None  # stats.ttest_ind(grupo_A, grupo_B)
    alpha = 0.05  # Nivel de significancia
    
    print(f"Media Grupo A: {media_A:.2f}")
    print(f"Media Grupo B: {media_B:.2f}")
    print(f"Estadístico t: {t_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print("✅ Rechazamos H0: Hay diferencia significativa")
    else:
        print("❌ No rechazamos H0: No hay diferencia significativa")
    
    # Visualización
    plt.figure(figsize=(10, 6))
    plt.boxplot([grupo_A, grupo_B], labels=['Grupo A', 'Grupo B'])
    plt.ylabel('Calificación')
    plt.title(f'Comparación de Grupos (p={p_value:.4f})')
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# EJERCICIO 4: Intervalo de Confianza
# ============================================================================

def intervalo_confianza():
    """
    Calcula intervalo de confianza para la media.
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 4: Intervalo de Confianza")
    print("=" * 60)
    
    # Muestra de alturas (cm)
    alturas = np.array([165, 170, 168, 172, 169, 171, 167, 173, 
                       170, 168, 172, 169, 171, 170, 168])
    
    # TODO: Calcula estadísticas
    n = len(alturas)
    media = None
    std_error = None  # stats.sem(alturas)
    
    # TODO: Intervalo de confianza del 95%
    confidence = 0.95
    intervalo = None  # stats.t.interval(confidence, n-1, media, std_error)
    
    print(f"Media: {media:.2f} cm")
    print(f"IC 95%: [{intervalo[0]:.2f}, {intervalo[1]:.2f}]")
    print(f"Interpretación: Con 95% de confianza, la media poblacional")
    print(f"está entre {intervalo[0]:.2f} y {intervalo[1]:.2f} cm")


# ============================================================================
# EJERCICIO 5: Chi-Cuadrado (Variables Categóricas)
# ============================================================================

def prueba_chi_cuadrado():
    """
    Prueba chi-cuadrado para independencia.
    
    ¿Existe relación entre género y preferencia de lenguaje?
    """
    print("\n" + "=" * 60)
    print("EJERCICIO 5: Prueba Chi-Cuadrado")
    print("=" * 60)
    
    # Tabla de contingencia
    #                Python  Java  JavaScript
    # Hombres           30     20      15
    # Mujeres           25     15      20
    
    tabla = np.array([
        [30, 20, 15],  # Hombres
        [25, 15, 20]   # Mujeres
    ])
    
    # TODO: Realiza prueba chi-cuadrado
    chi2, p_value, dof, expected = None, None, None, None
    # stats.chi2_contingency(tabla)
    
    print("Tabla observada:")
    print(tabla)
    print("\nTabla esperada (si son independientes):")
    print(expected)
    print(f"\nChi-cuadrado: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("✅ Hay relación entre género y preferencia")
    else:
        print("❌ No hay evidencia de relación")


# ============================================================================
# PROYECTO: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

def proyecto_eda():
    """
    Análisis exploratorio completo de un dataset.
    """
    print("\n" + "=" * 60)
    print("PROYECTO: Análisis Exploratorio de Datos")
    print("=" * 60)
    
    # Genera dataset sintético
    np.random.seed(42)
    n = 200
    
    data = pd.DataFrame({
        'edad': np.random.randint(18, 65, n),
        'experiencia': np.random.randint(0, 30, n),
        'educacion': np.random.choice(['Pregrado', 'Maestría', 'Doctorado'], n),
        'salario': np.random.normal(50000, 15000, n)
    })
    
    # Añade correlación: salario ~ experiencia
    data['salario'] = 30000 + data['experiencia'] * 1500 + np.random.normal(0, 5000, n)
    
    print("\n1️⃣ Primeras filas:")
    print(data.head())
    
    print("\n2️⃣ Estadísticas descriptivas:")
    print(data.describe())
    
    print("\n3️⃣ Valores nulos:")
    print(data.isnull().sum())
    
    # TODO: Análisis de correlación
    correlacion_numerica = None  # data[['edad', 'experiencia', 'salario']].corr()
    
    print("\n4️⃣ Matriz de correlación:")
    print(correlacion_numerica)
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histograma de salarios
    axes[0, 0].hist(data['salario'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribución de Salarios')
    axes[0, 0].set_xlabel('Salario')
    axes[0, 0].set_ylabel('Frecuencia')
    
    # Boxplot por educación
    data.boxplot(column='salario', by='educacion', ax=axes[0, 1])
    axes[0, 1].set_title('Salario por Educación')
    
    # Scatter: Experiencia vs Salario
    axes[1, 0].scatter(data['experiencia'], data['salario'], alpha=0.6)
    axes[1, 0].set_title('Experiencia vs Salario')
    axes[1, 0].set_xlabel('Años de Experiencia')
    axes[1, 0].set_ylabel('Salario')
    
    # Heatmap de correlación
    import seaborn as sns
    sns.heatmap(correlacion_numerica, annot=True, cmap='coolwarm', 
                ax=axes[1, 1], vmin=-1, vmax=1)
    axes[1, 1].set_title('Mapa de Calor - Correlación')
    
    plt.tight_layout()
    plt.show()
    
    # TODO: Responde preguntas:
    # - ¿Cuál es el salario promedio?
    # - ¿Qué nivel educativo tiene mayor salario?
    # - ¿Existe correlación entre experiencia y salario?


# ============================================================================
# PROYECTO: DETECCIÓN DE OUTLIERS
# ============================================================================

def deteccion_outliers():
    """
    Identifica valores atípicos en un dataset.
    """
    print("\n" + "=" * 60)
    print("PROYECTO: Detección de Outliers")
    print("=" * 60)
    
    # Datos con algunos outliers
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(100, 15, 100),  # Datos normales
        [200, 210, 5, -10]                # Outliers
    ])
    
    # TODO: Método 1 - IQR (Rango Intercuartílico)
    Q1 = None
    Q3 = None
    IQR = None
    
    lower_bound = None  # Q1 - 1.5*IQR
    upper_bound = None  # Q3 + 1.5*IQR
    
    outliers_iqr = None  # data[(data < lower_bound) | (data > upper_bound)]
    
    print(f"Método IQR:")
    print(f"  Límite inferior: {lower_bound:.2f}")
    print(f"  Límite superior: {upper_bound:.2f}")
    print(f"  Outliers encontrados: {len(outliers_iqr)}")
    
    # TODO: Método 2 - Z-score
    z_scores = None  # np.abs(stats.zscore(data))
    threshold = 3
    outliers_z = None  # data[z_scores > threshold]
    
    print(f"\nMétodo Z-score (|z| > {threshold}):")
    print(f"  Outliers encontrados: {len(outliers_z)}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Boxplot
    axes[0].boxplot(data)
    axes[0].set_title('Boxplot - Detección IQR')
    axes[0].set_ylabel('Valor')
    
    # Histograma con límites
    axes[1].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(lower_bound, color='r', linestyle='--', 
                    label=f'Límite inferior: {lower_bound:.1f}')
    axes[1].axvline(upper_bound, color='r', linestyle='--',
                    label=f'Límite superior: {upper_bound:.1f}')
    axes[1].set_title('Distribución con Límites IQR')
    axes[1].set_xlabel('Valor')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# TESTS
# ============================================================================

def run_tests():
    """Verifica comprensión estadística"""
    print("\n🧪 Tests de estadística...\n")
    
    # Test 1: Media
    data = np.array([1, 2, 3, 4, 5])
    assert np.mean(data) == 3, "Error en media"
    print("✅ Test 1: Media correcta")
    
    # Test 2: Desviación estándar
    assert abs(np.std(data, ddof=1) - 1.5811) < 0.001, "Error en std"
    print("✅ Test 2: Desviación estándar correcta")
    
    # Test 3: Correlación
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    corr = np.corrcoef(x, y)[0, 1]
    assert abs(corr - 1.0) < 0.001, "Error en correlación"
    print("✅ Test 3: Correlación correcta")
    
    print("\n🎉 ¡Tests pasados!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ESTADÍSTICA - EJERCICIOS PRÁCTICOS")
    print("=" * 60)
    
    # Descomenta para ejecutar:
    
    # estadisticas_descriptivas()
    # analisis_correlacion()
    # prueba_t_student()
    # intervalo_confianza()
    # prueba_chi_cuadrado()
    
    # Proyectos
    # proyecto_eda()
    # deteccion_outliers()
    
    # Tests
    # run_tests()
    
    print("\n📚 Revisa ../README.md para teoría completa")
    print("💡 Estos ejercicios son fundamentales para ML!")
