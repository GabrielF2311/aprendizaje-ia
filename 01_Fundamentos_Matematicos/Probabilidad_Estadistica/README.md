# 🎲 Probabilidad y Estadística - Semanas 7-10

## 🎯 Objetivos del Módulo

- Dominar conceptos probabilísticos fundamentales
- Entender distribuciones de probabilidad
- Aplicar estadística descriptiva e inferencial
- Usar herramientas estadísticas para ML

**Duración**: 4 semanas (Probabilidad: 2 semanas, Estadística: 2 semanas)

---

## 📂 Estructura

```
Probabilidad_Estadistica/
├── README.md (este archivo)
├── probabilidad/              # Semanas 7-8
│   ├── fundamentos/
│   ├── distribuciones/
│   └── proyecto_monte_carlo/
└── estadistica/               # Semanas 9-10
    ├── descriptiva/
    ├── inferencial/
    └── proyecto_analisis/
```

---

## PARTE 1: PROBABILIDAD (Semanas 7-8)

### 🎲 Semana 7: Fundamentos de Probabilidad

#### Día 1: Probabilidad Básica
- Experimentos aleatorios
- Espacio muestral
- Eventos y probabilidad
- Axiomas de probabilidad
- Probabilidad condicional

#### Día 2: Teorema de Bayes
- Probabilidad conjunta
- Probabilidad condicional
- Teorema de Bayes
- Aplicaciones: clasificadores Bayesianos

#### Día 3: Variables Aleatorias
- Discretas vs continuas
- Función de masa/densidad de probabilidad
- Función de distribución acumulativa
- Esperanza (valor esperado)
- Varianza y desviación estándar

### 🎲 Semana 8: Distribuciones de Probabilidad

#### Día 4: Distribuciones Discretas
- Bernoulli (ensayo único)
- Binomial (n ensayos)
- Poisson (eventos en tiempo)
- Geométrica

#### Día 5-6: Distribuciones Continuas
- Uniforme
- Normal (Gaussiana) ⭐
- Exponencial
- Beta, Gamma

#### Día 7: Proyecto - Simulaciones Monte Carlo
- Simular procesos aleatorios
- Estimar π con Monte Carlo
- Aplicaciones en IA

---

## PARTE 2: ESTADÍSTICA (Semanas 9-10)

### 📊 Semana 9: Estadística Descriptiva e Inferencial

#### Día 1: Estadística Descriptiva
- Medidas de tendencia central (media, mediana, moda)
- Medidas de dispersión (varianza, std, rango)
- Percentiles y quartiles
- Visualización de datos

#### Día 2: Muestreo
- Población vs muestra
- Tipos de muestreo
- Error estándar
- Teorema del límite central

#### Día 3: Intervalos de Confianza
- Estimación puntual vs intervalo
- Intervalos de confianza
- Nivel de confianza (95%, 99%)

### 📊 Semana 10: Correlación y Análisis

#### Día 4: Pruebas de Hipótesis
- Hipótesis nula vs alternativa
- p-value
- Errores tipo I y II
- t-test, z-test

#### Día 5: Correlación y Covarianza
- Covarianza
- Correlación de Pearson
- Correlación de Spearman
- Interpretación

#### Día 6-7: Proyecto - Análisis Estadístico Completo
- EDA estadístico de dataset
- Pruebas de hipótesis
- Correlaciones
- Conclusiones

---

## 📚 Conceptos Clave - PROBABILIDAD

### Probabilidad Básica

$$P(A) = \frac{\text{Casos favorables}}{\text{Casos totales}}$$

**Propiedades**:
- $0 \leq P(A) \leq 1$
- $P(\Omega) = 1$ (espacio muestral)
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### Probabilidad Condicional

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Lectura**: "Probabilidad de A dado que B ocurrió"

### Teorema de Bayes

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Aplicación en IA**: Clasificadores Naive Bayes

```python
# Ejemplo: Spam detection
# P(spam | contiene "gratis") = ?

P_spam = 0.3  # Prior: 30% de emails son spam
P_gratis_dado_spam = 0.8  # Likelihood
P_gratis = 0.35  # Evidence

P_spam_dado_gratis = (P_gratis_dado_spam * P_spam) / P_gratis
print(f"P(spam | 'gratis') = {P_spam_dado_gratis:.2%}")
```

### Esperanza y Varianza

**Esperanza** (valor esperado):
$$E[X] = \sum x_i \cdot P(x_i)$$ (discreta)
$$E[X] = \int x \cdot f(x) dx$$ (continua)

**Varianza**:
$$Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

### Distribución Normal

La más importante en estadística:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$

Propiedades:
- Simétrica alrededor de $\mu$ (media)
- 68% dentro de 1 std
- 95% dentro de 2 std
- 99.7% dentro de 3 std

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Crear distribución normal
mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, mu, sigma)

plt.plot(x, y)
plt.title('Distribución Normal Estándar')
plt.show()
```

---

## 📚 Conceptos Clave - ESTADÍSTICA

### Medidas de Tendencia Central

```python
import numpy as np

data = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]

media = np.mean(data)        # Promedio
mediana = np.median(data)    # Valor central
moda = stats.mode(data)      # Más frecuente
```

### Medidas de Dispersión

```python
varianza = np.var(data)      # Promedio de desviaciones²
std = np.std(data)           # Raíz de varianza
rango = np.max(data) - np.min(data)
```

### Teorema del Límite Central

**Idea clave**: La media de muestras grandes tiende a distribuirse normalmente, sin importar la distribución original.

```python
# Demostración
samples = [np.mean(np.random.exponential(size=50)) for _ in range(1000)]
plt.hist(samples, bins=30)
plt.title('Distribución de medias muestrales - Se ve Normal!')
plt.show()
```

### Intervalos de Confianza

**Intervalo del 95%**:
$$[\bar{x} - 1.96\frac{s}{\sqrt{n}}, \bar{x} + 1.96\frac{s}{\sqrt{n}}]$$

```python
from scipy import stats

# Calcular IC 95%
confidence = 0.95
data = np.random.normal(100, 15, 50)
ci = stats.t.interval(confidence, len(data)-1, 
                       loc=np.mean(data), 
                       scale=stats.sem(data))
print(f"IC 95%: {ci}")
```

### Pruebas de Hipótesis

**Proceso**:
1. Formular $H_0$ (hipótesis nula) y $H_1$ (alternativa)
2. Elegir nivel de significancia ($\alpha = 0.05$)
3. Calcular estadístico de prueba
4. Calcular p-value
5. Decisión: Si p-value < $\alpha$, rechazar $H_0$

```python
# t-test: ¿Las medias son diferentes?
group_a = [23, 25, 27, 24, 26]
group_b = [30, 32, 31, 33, 29]

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("Rechazamos H0: Las medias SON diferentes")
else:
    print("No rechazamos H0: No hay evidencia de diferencia")
```

### Correlación

**Correlación de Pearson** (lineal):
$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

Rango: $[-1, 1]$
- $r = 1$: Correlación positiva perfecta
- $r = 0$: Sin correlación
- $r = -1$: Correlación negativa perfecta

```python
# Calcular correlación
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlación: {correlation:.3f}")

# Visualizar
plt.scatter(x, y)
plt.title(f'Correlación: {correlation:.3f}')
plt.show()
```

---

## 💻 Proyectos

### Proyecto 1: Simulador Monte Carlo

**Objetivo**: Usar simulaciones para resolver problemas

**Tareas**:
1. Estimar π lanzando puntos aleatorios
2. Simular el problema de Monty Hall
3. Calcular probabilidad de ganar la lotería
4. Simular random walks

### Proyecto 2: Análisis Estadístico Completo

**Objetivo**: Análisis exhaustivo de un dataset

**Tareas**:
1. Cargar dataset (Kaggle, UCI ML Repository)
2. Estadística descriptiva completa
3. Visualizaciones (histogramas, boxplots, scatter)
4. Pruebas de normalidad
5. Correlaciones entre variables
6. Pruebas de hipótesis
7. Conclusiones y reportes

---

## ✅ Checklist Completo

### Probabilidad
- [ ] Entiendo probabilidad básica y condicional
- [ ] Puedo aplicar teorema de Bayes
- [ ] Conozco variables aleatorias y esperanza
- [ ] Domino distribuciones (Binomial, Normal, Poisson)
- [ ] Implementé simulaciones Monte Carlo

### Estadística
- [ ] Calculo medidas descriptivas
- [ ] Entiendo muestreo y CLT
- [ ] Construyo intervalos de confianza
- [ ] Realizo pruebas de hipótesis
- [ ] Interpreto correlaciones
- [ ] Completé análisis estadístico de dataset

---

## 🔗 Conexión con Machine Learning

### Probabilidad en ML

**Clasificación Probabilística**:
```python
# Logistic Regression predice P(y=1|x)
# Naive Bayes usa Teorema de Bayes
# GANs modelan distribuciones de datos
```

**Regularización Bayesiana**:
```python
# Ridge Regression = Gaussian prior
# Lasso = Laplacian prior
```

### Estadística en ML

**Validación de Modelos**:
- t-test para comparar modelos
- Intervalos de confianza para métricas
- Correlación para feature selection

**A/B Testing**:
- Pruebas de hipótesis para experimentos
- Determinar si un cambio tiene efecto

---

## 📚 Recursos

### Libros
- *Introduction to Probability* - Blitzstein & Hwang
- *Statistics* - Freedman, Pisani, Purves
- *Think Stats* - Allen Downey (Python)

### Videos
- **StatQuest**: Todos los videos de estadística
- **Khan Academy**: Probability and Statistics
- **3Blue1Brown**: Bayesian Theorem

### Prácticas
- [Seeing Theory](https://seeing-theory.brown.edu/) - Visualizaciones interactivas
- [Probability & Statistics Cookbook](http://statistics.zone/)

---

**¡La probabilidad y estadística son fundamentales para entender ML!** 🎲📊

**Siguiente Módulo**: Python para IA
