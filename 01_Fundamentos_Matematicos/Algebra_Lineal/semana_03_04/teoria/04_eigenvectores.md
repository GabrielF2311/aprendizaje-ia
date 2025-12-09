# Día 3-4: Eigenvalores y Eigenvectores

## 📋 Objetivos
- Comprender el concepto de eigenvalores y eigenvectores
- Calcular eigenvalores y eigenvectores de matrices
- Entender la diagonalización de matrices
- Aplicar eigenvalues en problemas de Machine Learning
- Interpretar geométricamente eigenvalores y eigenvectores

---

## 1. Fundamentos

### 1.1 Definición

Para una matriz cuadrada **A** de tamaño $n \times n$, un **eigenvector** $\mathbf{v}$ (no nulo) y su correspondiente **eigenvalor** $\lambda$ satisfacen:

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

**Interpretación:**
- La transformación **A** solo **escala** el vector $\mathbf{v}$ por el factor $\lambda$
- La **dirección** de $\mathbf{v}$ no cambia (solo su magnitud)
- $\mathbf{v}$ es una "dirección especial" para la transformación **A**

### 1.2 Nombres Alternativos

| Español | Inglés | Alemán (original) |
|---------|--------|-------------------|
| Valores propios | Eigenvalues | Eigenwerte |
| Vectores propios | Eigenvectors | Eigenvektoren |
| Autovalores | - | - |
| Autovectores | - | - |

### 1.3 Ejemplo Simple

$$
A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

**Verificar:**
$$
A\mathbf{v} = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix} = 3 \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 3\mathbf{v}
$$

Por lo tanto, $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ es un eigenvector con eigenvalor $\lambda = 3$.

---

## 2. Cálculo de Eigenvalores

### 2.1 Ecuación Característica

Para encontrar eigenvalores, resolvemos:

$$
\det(A - \lambda I) = 0
$$

Esta es la **ecuación característica**. El **polinomio característico** es $\det(A - \lambda I)$.

### 2.2 Ejemplo Paso a Paso

**Matriz:**
$$
A = \begin{bmatrix} 4 & -2 \\ 1 & 1 \end{bmatrix}
$$

**Paso 1:** Formar $A - \lambda I$
$$
A - \lambda I = \begin{bmatrix} 4 & -2 \\ 1 & 1 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 4-\lambda & -2 \\ 1 & 1-\lambda \end{bmatrix}
$$

**Paso 2:** Calcular determinante
$$
\det(A - \lambda I) = (4-\lambda)(1-\lambda) - (-2)(1)
$$
$$
= 4 - 4\lambda - \lambda + \lambda^2 + 2
$$
$$
= \lambda^2 - 5\lambda + 6
$$

**Paso 3:** Resolver ecuación característica
$$
\lambda^2 - 5\lambda + 6 = 0
$$
$$
(\lambda - 2)(\lambda - 3) = 0
$$

**Eigenvalores:** $\lambda_1 = 2, \quad \lambda_2 = 3$

### 2.3 Propiedades de Eigenvalores

**Para una matriz $A$ de $n \times n$:**

1. **Suma de eigenvalores = Traza:**
   $$\sum_{i=1}^{n} \lambda_i = \text{tr}(A) = \sum_{i=1}^{n} a_{ii}$$

2. **Producto de eigenvalores = Determinante:**
   $$\prod_{i=1}^{n} \lambda_i = \det(A)$$

3. Una matriz es **invertible** ⟺ todos sus eigenvalores son $\neq 0$

4. Los eigenvalores de $A^T$ son los mismos que los de $A$

5. Los eigenvalores de $A^{-1}$ son $\frac{1}{\lambda_i}$

6. Los eigenvalores de $A^k$ son $\lambda_i^k$

---

## 3. Cálculo de Eigenvectores

### 3.1 Proceso

Una vez encontrado un eigenvalor $\lambda$, calculamos su eigenvector resolviendo:

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

Este es un **sistema homogéneo** de ecuaciones lineales.

### 3.2 Ejemplo Completo

Usando $A = \begin{bmatrix} 4 & -2 \\ 1 & 1 \end{bmatrix}$ con $\lambda_1 = 2$:

**Paso 1:** Formar $(A - 2I)$
$$
A - 2I = \begin{bmatrix} 2 & -2 \\ 1 & -1 \end{bmatrix}
$$

**Paso 2:** Resolver $(A - 2I)\mathbf{v} = \mathbf{0}$
$$
\begin{bmatrix} 2 & -2 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

Ecuaciones:
$$
\begin{cases}
2v_1 - 2v_2 = 0 \\
v_1 - v_2 = 0
\end{cases}
$$

Ambas ecuaciones dan: $v_1 = v_2$

**Paso 3:** Expresar solución
$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_1 \end{bmatrix} = v_1 \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

**Eigenvector (normalizado):** $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

### 3.3 Para $\lambda_2 = 3$

$$
A - 3I = \begin{bmatrix} 1 & -2 \\ 1 & -2 \end{bmatrix}
$$

Resolver:
$$
v_1 - 2v_2 = 0 \quad \Rightarrow \quad v_1 = 2v_2
$$

**Eigenvector:** $\mathbf{v}_2 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

### 3.4 Verificación

**Para $\lambda_1 = 2, \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$:**
$$
A\mathbf{v}_1 = \begin{bmatrix} 4 & -2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix} = 2 \begin{bmatrix} 1 \\ 1 \end{bmatrix} \quad ✓
$$

**Para $\lambda_2 = 3, \mathbf{v}_2 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$:**
$$
A\mathbf{v}_2 = \begin{bmatrix} 4 & -2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 6 \\ 3 \end{bmatrix} = 3 \begin{bmatrix} 2 \\ 1 \end{bmatrix} \quad ✓
$$

---

## 4. Interpretación Geométrica

### 4.1 Transformación Lineal

En 2D, una matriz $A$ transforma vectores del plano:

```
Original        →    Transformado
   ↑                     ↗
   |                    /
   |                   /
   • ----→         • ----→
```

Los **eigenvectores** son direcciones que **solo se escalan**, no rotan:

```
Eigenvector (λ > 1):  Se estira
────────→  →  ────────────────→

Eigenvector (0 < λ < 1):  Se encoge
────────→  →  ──→

Eigenvector (λ < 0):  Se invierte y escala
────────→  →  ←──────
```

### 4.2 Ejemplo Visual

**Matriz de escalado:**
$$
A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}
$$

- $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (eje x): se escala por 3
- $\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (eje y): se escala por 2

### 4.3 Eigenvalores Complejos

Si una matriz tiene eigenvalores **complejos conjugados**, representa una **rotación + escalado**.

**Ejemplo:**
$$
A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \quad \text{(rotación 90°)}
$$

Eigenvalores: $\lambda = \pm i$ (imaginarios puros)

---

## 5. Diagonalización

### 5.1 Definición

Una matriz $A$ es **diagonalizable** si existe una matriz invertible $P$ tal que:

$$
A = PDP^{-1}
$$

Donde **D** es una matriz diagonal de eigenvalores:
$$
D = \begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$

Y **P** tiene eigenvectores como columnas:
$$
P = [\mathbf{v}_1 \, | \, \mathbf{v}_2 \, | \, \cdots \, | \, \mathbf{v}_n]
$$

### 5.2 Condiciones para Diagonalización

Una matriz $n \times n$ es diagonalizable si:
- Tiene **n eigenvectores linealmente independientes**, O
- Todos los eigenvalores son **distintos** (suficiente pero no necesario)

### 5.3 Ejemplo Completo

**Matriz:**
$$
A = \begin{bmatrix} 4 & -2 \\ 1 & 1 \end{bmatrix}
$$

**Ya calculamos:**
- $\lambda_1 = 2, \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- $\lambda_2 = 3, \mathbf{v}_2 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

**Formar P y D:**
$$
P = \begin{bmatrix} 1 & 2 \\ 1 & 1 \end{bmatrix}, \quad
D = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}
$$

**Calcular $P^{-1}$:**
$$
P^{-1} = \frac{1}{\det(P)} \begin{bmatrix} 1 & -2 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} -1 & 2 \\ 1 & -1 \end{bmatrix}
$$

**Verificar $A = PDP^{-1}$:**
$$
PDP^{-1} = \begin{bmatrix} 1 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} -1 & 2 \\ 1 & -1 \end{bmatrix}
$$
$$
= \begin{bmatrix} 1 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} -2 & 4 \\ 3 & -3 \end{bmatrix} = \begin{bmatrix} 4 & -2 \\ 1 & 1 \end{bmatrix} = A \quad ✓
$$

### 5.4 Ventajas de la Diagonalización

**1. Potencias de matrices:**
$$
A^k = (PDP^{-1})^k = PD^kP^{-1}
$$

Donde:
$$
D^k = \begin{bmatrix} \lambda_1^k & 0 \\ 0 & \lambda_2^k \end{bmatrix}
$$

**Mucho más fácil** que calcular $A \times A \times A \times \cdots$

**2. Ecuaciones diferenciales:**
Resolver $\frac{d\mathbf{x}}{dt} = A\mathbf{x}$ se simplifica con diagonalización.

**3. Análisis de estabilidad:**
Un sistema es estable si todos $|\lambda_i| < 1$

---

## 6. Matrices Especiales

### 6.1 Matrices Simétricas

Si $A = A^T$ (simétrica):
- Todos los eigenvalores son **reales**
- Eigenvectores son **ortogonales** entre sí
- Siempre es **diagonalizable**

**Ejemplo:**
$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

Eigenvalores: $\lambda_1 = 3, \lambda_2 = 1$

Eigenvectores: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

Verificar ortogonalidad: $\mathbf{v}_1 \cdot \mathbf{v}_2 = 1 \times 1 + 1 \times (-1) = 0$ ✓

### 6.2 Matrices Positivas Definidas

Una matriz simétrica $A$ es **positiva definida** si:
- Todos sus eigenvalores son $\lambda_i > 0$

**Propiedades:**
- $\mathbf{x}^T A \mathbf{x} > 0$ para todo $\mathbf{x} \neq \mathbf{0}$
- Tiene inversa
- Importante en optimización (Hessiana)

### 6.3 Matrices Ortogonales

Una matriz $Q$ es **ortogonal** si $Q^T Q = I$:
- Eigenvalores tienen magnitud 1: $|\lambda_i| = 1$
- Preserva longitudes y ángulos
- Representa rotaciones/reflexiones

---

## 7. Implementación en Python

### 7.1 Calcular Eigenvalores y Eigenvectores

```python
import numpy as np

# Matriz ejemplo
A = np.array([[4, -2],
              [1,  1]])

# Calcular eigenvalores y eigenvectores
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalores:")
print(eigenvalues)
# [2. 3.]

print("\nEigenvectores (columnas):")
print(eigenvectors)
# [[0.70710678 0.89442719]
#  [0.70710678 0.4472136 ]]

# Normalización: NumPy devuelve eigenvectores normalizados (norma = 1)
print("\nNorma del primer eigenvector:")
print(np.linalg.norm(eigenvectors[:, 0]))  # 1.0
```

### 7.2 Verificar Resultados

```python
# Para cada par eigenvalor-eigenvector
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    
    # Av = λv
    left_side = A @ v_i
    right_side = lambda_i * v_i
    
    print(f"\nEigenvalor {i+1}: λ = {lambda_i:.2f}")
    print(f"A @ v = {left_side}")
    print(f"λ @ v = {right_side}")
    print(f"¿Iguales? {np.allclose(left_side, right_side)}")
```

### 7.3 Diagonalización

```python
# Matrices P y D
P = eigenvectors
D = np.diag(eigenvalues)

print("Matriz P (eigenvectores):")
print(P)

print("\nMatriz D (eigenvalores):")
print(D)

# Reconstruir A = P @ D @ P^(-1)
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv

print("\nA reconstruida:")
print(A_reconstructed)

print("\n¿Coincide con A original?")
print(np.allclose(A, A_reconstructed))  # True
```

### 7.4 Potencias de Matrices

```python
# Calcular A^10 eficientemente

# Método tradicional (lento para k grande)
A_10_traditional = np.linalg.matrix_power(A, 10)

# Método con diagonalización
D_10 = np.diag(eigenvalues ** 10)
A_10_diag = P @ D_10 @ P_inv

print("A^10 (tradicional):")
print(A_10_traditional)

print("\nA^10 (diagonalización):")
print(A_10_diag)

print("\n¿Iguales?")
print(np.allclose(A_10_traditional, A_10_diag))
```

### 7.5 Eigenvalores de Matrices Grandes

```python
# Para matrices grandes, usar eigenvalores específicos
from scipy.sparse.linalg import eigsh

# Matriz grande simétrica
n = 1000
A_large = np.random.randn(n, n)
A_large = (A_large + A_large.T) / 2  # Hacerla simétrica

# Calcular solo los 5 eigenvalores más grandes
k = 5
eigenvalues_top5, eigenvectors_top5 = eigsh(A_large, k=k, which='LM')

print(f"Top {k} eigenvalores:")
print(eigenvalues_top5)
```

---

## 8. Aplicaciones en Machine Learning

### 8.1 PCA (Principal Component Analysis)

**Objetivo:** Reducir dimensionalidad encontrando direcciones de máxima varianza.

**Proceso:**
1. Centrar datos: $X_{\text{centered}} = X - \text{mean}(X)$
2. Calcular matriz de covarianza: $C = \frac{1}{n}X^T X$
3. Calcular eigenvalores y eigenvectores de $C$
4. Los eigenvectores con **mayores eigenvalores** son las **componentes principales**

```python
import numpy as np

# Datos: 100 muestras, 5 características
X = np.random.randn(100, 5)

# Centrar
X_centered = X - np.mean(X, axis=0)

# Matriz de covarianza
cov_matrix = (X_centered.T @ X_centered) / len(X)

# Eigenvalores y eigenvectores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ordenar por eigenvalor (descendente)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Varianza explicada por cada componente:")
print(eigenvalues / eigenvalues.sum())

# Proyectar a 2 componentes principales
k = 2
principal_components = eigenvectors[:, :k]
X_reduced = X_centered @ principal_components

print(f"\nDimensión original: {X.shape}")
print(f"Dimensión reducida: {X_reduced.shape}")
```

### 8.2 PageRank (Google)

**Problema:** Rankear páginas web según su importancia.

**Solución:** El ranking es el **eigenvector principal** de la matriz de transición web.

$$
\mathbf{r} = A\mathbf{r}
$$

Donde $\mathbf{r}$ es el vector de rankings (eigenvalor $\lambda = 1$).

### 8.3 Análisis de Estabilidad de Redes Neuronales

En RNNs, la estabilidad depende de eigenvalores de la matriz de pesos:
- Si $|\lambda_{\max}| > 1$: **Exploding gradients**
- Si $|\lambda_{\max}| < 1$: **Vanishing gradients**

### 8.4 Spectral Clustering

**Agrupamiento basado en eigenvectores de la matriz Laplaciana del grafo:**

1. Construir matriz de similitud $W$
2. Calcular matriz Laplaciana: $L = D - W$
3. Eigenvalores y eigenvectores de $L$
4. Usar primeros $k$ eigenvectores para clustering

### 8.5 Análisis de Componentes Independientes (ICA)

Generalización de PCA que busca componentes **estadísticamente independientes** (no solo ortogonales).

---

## 9. Casos Especiales

### 9.1 Eigenvalores Repetidos (Multiplicidad)

**Multiplicidad algebraica:** Veces que $\lambda$ aparece en el polinomio característico

**Multiplicidad geométrica:** Número de eigenvectores linealmente independientes

**Ejemplo:**
$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}
$$

- $\lambda = 2$ (doble)
- Eigenvectores: cualquier vector $\begin{bmatrix} x \\ y \end{bmatrix}$ (2D de soluciones)

### 9.2 Matrices No Diagonalizables

**Ejemplo:**
$$
A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
$$

- $\lambda = 1$ (doble)
- Solo 1 eigenvector independiente: $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$
- **No diagonalizable** (forma de Jordan)

### 9.3 Eigenvalores Complejos

**Ejemplo:**
$$
A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

Eigenvalores: $\lambda = i, -i$

Eigenvectores: complejos

**Representan rotación pura (90° en este caso)**

---

## 10. Errores Comunes

### ❌ Error 1: Confundir Eigenvalor con Eigenvector

```python
# ❌ Incorrecto
eigenvalue = eigenvectors[0]  # Esto es un vector, no un valor

# ✅ Correcto
lambda_1 = eigenvalues[0]      # Eigenvalor (escalar)
v_1 = eigenvectors[:, 0]       # Eigenvector (vector columna)
```

### ❌ Error 2: No Normalizar Eigenvectores

Los eigenvectores tienen **dirección** importante, pero **magnitud arbitraria**.

NumPy normaliza (norma = 1), pero soluciones manuales pueden tener cualquier escala.

### ❌ Error 3: Asumir Diagonalización Siempre Posible

No todas las matrices son diagonalizables. Verificar primero.

### ❌ Error 4: Ignorar Orden de Eigenvalores

En PCA, el **orden importa**: mayores eigenvalores = mayor varianza.

```python
# Siempre ordenar
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

---

## 11. Ejercicios Prácticos

### Ejercicio 1: Cálculo Manual
Encuentra eigenvalores y eigenvectores de:
$$
A = \begin{bmatrix} 5 & 2 \\ 2 & 5 \end{bmatrix}
$$

### Ejercicio 2: Verificación
Verifica que $A\mathbf{v} = \lambda\mathbf{v}$ para tus resultados del ejercicio 1.

### Ejercicio 3: Potencia de Matriz
Calcula $A^{20}$ usando diagonalización para:
$$
A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}
$$

### Ejercicio 4: PCA Simple
Implementa PCA desde cero en un dataset 2D:
```python
# Generar datos correlacionados
X = np.random.randn(100, 2) @ [[2, 0], [0, 0.5]]
# Aplicar PCA y visualizar
```

### Ejercicio 5: Eigenvalores de Matrices Especiales
Encuentra los eigenvalores de:
- Matriz identidad $I$
- Matriz de ceros
- Matriz triangular $\begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{bmatrix}$

---

## 12. Recursos Adicionales

### 📺 Videos
- **3Blue1Brown:** "Eigenvectors and eigenvalues" (visualización excelente)
- **MIT OCW:** Gilbert Strang - Lecture 21

### 📚 Lecturas
- **Linear Algebra Done Right** (Axler): Capítulo 5
- **Deep Learning Book** (Goodfellow): Sección 2.7

### 🔧 Herramientas
- **GeoGebra:** Visualizar transformaciones y eigenvectores
- **NumPy/SciPy:** Cálculo numérico
- **SymPy:** Cálculo simbólico

---

## 📌 Resumen Clave

| Concepto | Fórmula/Propiedad |
|----------|-------------------|
| **Definición** | $A\mathbf{v} = \lambda\mathbf{v}$ |
| **Ec. característica** | $\det(A - \lambda I) = 0$ |
| **Eigenvector** | $(A - \lambda I)\mathbf{v} = \mathbf{0}$ |
| **Diagonalización** | $A = PDP^{-1}$ |
| **Suma eigenvalores** | $\sum \lambda_i = \text{tr}(A)$ |
| **Producto eigenvalores** | $\prod \lambda_i = \det(A)$ |
| **Python** | `np.linalg.eig(A)` |

---

## 🎯 Próximo Tema

**Día 5-6:** Descomposición en Valores Singulares (SVD)
- Generalización de eigenvalores a matrices no cuadradas
- Aplicaciones en compresión de imágenes
- Fundamento de muchos algoritmos de ML

---

*Los eigenvalores y eigenvectores son el corazón de PCA, análisis espectral y muchos otros algoritmos fundamentales de ML. ¡Domina este concepto!*
