# Día 5: Sistemas de Ecuaciones Lineales

## 📋 Objetivos del Día
- Comprender qué es un sistema de ecuaciones lineales
- Representar sistemas usando matrices
- Aplicar métodos de solución (eliminación gaussiana, matrices inversas)
- Identificar tipos de soluciones y su significado geométrico
- Aplicar sistemas lineales en problemas de Machine Learning

---

## 1. Fundamentos de Sistemas Lineales

### 1.1 Definición
Un **sistema de ecuaciones lineales** es un conjunto de ecuaciones de la forma:

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

**Donde:**
- $x_1, x_2, \ldots, x_n$ son las **incógnitas** (variables)
- $a_{ij}$ son los **coeficientes** (números conocidos)
- $b_i$ son los **términos independientes** (lado derecho)

### 1.2 Ejemplo Simple
Sistema de 2 ecuaciones con 2 incógnitas:

$$
\begin{cases}
2x + 3y = 8 \\
x - y = 1
\end{cases}
$$

**Solución:** $x = 2, \, y = \frac{4}{3}$

### 1.3 Representación Matricial
El mismo sistema se puede escribir como **Ax = b**:

$$
\underbrace{\begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix}}_{\text{Matriz A}}
\underbrace{\begin{bmatrix} x \\ y \end{bmatrix}}_{\text{Vector } \mathbf{x}}
=
\underbrace{\begin{bmatrix} 8 \\ 1 \end{bmatrix}}_{\text{Vector } \mathbf{b}}
$$

**Componentes:**
- **A:** Matriz de coeficientes ($m \times n$)
- **x:** Vector de incógnitas ($n \times 1$)
- **b:** Vector de términos independientes ($m \times 1$)

---

## 2. Tipos de Soluciones

### 2.1 Clasificación por Número de Soluciones

| Tipo | Descripción | Nombre Técnico |
|------|-------------|----------------|
| **Única** | Exactamente una solución | Sistema Compatible Determinado |
| **Infinitas** | Infinitas soluciones | Sistema Compatible Indeterminado |
| **Ninguna** | No hay solución | Sistema Incompatible |

### 2.2 Interpretación Geométrica (2D)

**Caso 1: Solución Única**
```
Las rectas se intersectan en UN punto
  y
  |
  |    /
  |   /
  |  X  (punto de intersección)
  | /  \
  |/____\_____ x
```

**Ejemplo:**
$$
\begin{cases}
x + y = 3 \\
x - y = 1
\end{cases}
\quad \Rightarrow \quad (x, y) = (2, 1)
$$

**Caso 2: Infinitas Soluciones**
```
Las rectas son COINCIDENTES (la misma línea)
  y
  |
  |   ////// (misma recta)
  |  //////
  | //////
  |/_________ x
```

**Ejemplo:**
$$
\begin{cases}
2x + 4y = 6 \\
x + 2y = 3
\end{cases}
\quad \Rightarrow \quad \text{Infinitas soluciones}
$$

**Caso 3: Sin Solución**
```
Las rectas son PARALELAS (nunca se intersectan)
  y
  |
  |   /////
  |  /////
  | --------- (paralela, más abajo)
  |/_________ x
```

**Ejemplo:**
$$
\begin{cases}
x + y = 2 \\
x + y = 5
\end{cases}
\quad \Rightarrow \quad \text{Sin solución}
$$

### 2.3 Interpretación Geométrica (3D+)

En 3 dimensiones, las ecuaciones representan **planos**:
- **Solución única:** Los planos se intersectan en un punto
- **Infinitas soluciones:** Los planos se intersectan en una línea o son el mismo plano
- **Sin solución:** Los planos son paralelos o no tienen intersección común

---

## 3. Métodos de Solución

### 3.1 Eliminación Gaussiana

El método más común para resolver sistemas. Convierte la matriz en **forma escalonada**.

**Ejemplo Completo:**

Sistema original:
$$
\begin{cases}
2x + y - z = 8 \\
-3x - y + 2z = -11 \\
-2x + y + 2z = -3
\end{cases}
$$

**Paso 1:** Matriz aumentada
$$
\left[\begin{array}{ccc|c}
2 & 1 & -1 & 8 \\
-3 & -1 & 2 & -11 \\
-2 & 1 & 2 & -3
\end{array}\right]
$$

**Paso 2:** Eliminar debajo del primer pivote
- $F_2 = F_2 + \frac{3}{2}F_1$
- $F_3 = F_3 + F_1$

$$
\left[\begin{array}{ccc|c}
2 & 1 & -1 & 8 \\
0 & 0.5 & 0.5 & 1 \\
0 & 2 & 1 & 5
\end{array}\right]
$$

**Paso 3:** Eliminar debajo del segundo pivote
- $F_3 = F_3 - 4F_2$

$$
\left[\begin{array}{ccc|c}
2 & 1 & -1 & 8 \\
0 & 0.5 & 0.5 & 1 \\
0 & 0 & -1 & 1
\end{array}\right]
$$

**Paso 4:** Sustitución hacia atrás
$$
\begin{align}
-z &= 1 \quad \Rightarrow \quad z = -1 \\
0.5y + 0.5(-1) &= 1 \quad \Rightarrow \quad y = 3 \\
2x + 3 - (-1) &= 8 \quad \Rightarrow \quad x = 2
\end{align}
$$

**Solución:** $(x, y, z) = (2, 3, -1)$

### 3.2 Eliminación de Gauss-Jordan

Extiende la eliminación gaussiana para obtener la **forma reducida escalonada**:

$$
\left[\begin{array}{ccc|c}
1 & 0 & 0 & x_{\text{sol}} \\
0 & 1 & 0 & y_{\text{sol}} \\
0 & 0 & 1 & z_{\text{sol}}
\end{array}\right]
$$

La solución se lee directamente.

### 3.3 Usando Matriz Inversa

Si **A** es invertible (cuadrada y det(A) ≠ 0):

$$
Ax = b \quad \Rightarrow \quad x = A^{-1}b
$$

**Ejemplo:**
$$
A = \begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix}, \quad
b = \begin{bmatrix} 8 \\ 1 \end{bmatrix}
$$

**Calcular $A^{-1}$:**
$$
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} -1 & -3 \\ -1 & 2 \end{bmatrix} = \frac{1}{-5} \begin{bmatrix} -1 & -3 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 0.2 & 0.6 \\ 0.2 & -0.4 \end{bmatrix}
$$

**Solución:**
$$
x = A^{-1}b = \begin{bmatrix} 0.2 & 0.6 \\ 0.2 & -0.4 \end{bmatrix} \begin{bmatrix} 8 \\ 1 \end{bmatrix} = \begin{bmatrix} 2.2 \\ 1.2 \end{bmatrix}
$$

⚠️ **Limitación:** Solo funciona si A es cuadrada e invertible.

### 3.4 Descomposición LU

Factoriza **A = LU** donde:
- **L:** Matriz triangular inferior (Lower)
- **U:** Matriz triangular superior (Upper)

**Ventaja:** Eficiente cuando se resuelven múltiples sistemas con la misma A.

---

## 4. Determinación de Soluciones

### 4.1 Rango de Matrices

El **rango** (rank) de una matriz es el número de filas/columnas linealmente independientes.

**Para sistema Ax = b:**

| Condición | Tipo de Solución |
|-----------|------------------|
| rango(A) = rango([A\|b]) = n | Solución única |
| rango(A) = rango([A\|b]) < n | Infinitas soluciones |
| rango(A) ≠ rango([A\|b]) | Sin solución |

Donde **n** es el número de incógnitas.

### 4.2 Determinante (Sistemas Cuadrados)

Para sistemas donde **A** es cuadrada ($m = n$):

| det(A) | Solución |
|--------|----------|
| det(A) ≠ 0 | Solución única |
| det(A) = 0 | Infinitas soluciones o ninguna |

---

## 5. Implementación en Python

### 5.1 Usando NumPy (Recomendado)

```python
import numpy as np

# Sistema: 2x + 3y = 8
#          x - y = 1

A = np.array([[2, 3],
              [1, -1]])
b = np.array([8, 1])

# Método 1: numpy.linalg.solve() - Más eficiente
x = np.linalg.solve(A, b)
print(f"Solución: x = {x[0]}, y = {x[1]}")
# Solución: x = 2.2, y = 1.2

# Método 2: Usando matriz inversa
A_inv = np.linalg.inv(A)
x = A_inv @ b
print(f"Solución (inversa): {x}")

# Verificar la solución
resultado = A @ x
print(f"Verificación A@x = {resultado}")
print(f"b original = {b}")
print(f"¿Correcto? {np.allclose(resultado, b)}")
```

### 5.2 Sistema 3×3

```python
import numpy as np

# Sistema:
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])

b = np.array([8, -11, -3])

# Resolver
x = np.linalg.solve(A, b)
print(f"Solución: x={x[0]:.2f}, y={x[1]:.2f}, z={x[2]:.2f}")
# Solución: x=2.00, y=3.00, z=-1.00

# Verificar determinante (debe ser ≠ 0)
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.2f}")  # Si ≈ 0, no hay solución única
```

### 5.3 Manejo de Sistemas Sin Solución o Indeterminados

```python
import numpy as np

# Sistema sin solución (rectas paralelas)
A = np.array([[1, 1],
              [1, 1]])
b = np.array([2, 5])

try:
    x = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    print("El sistema no tiene solución única")

# Verificar con determinante
print(f"det(A) = {np.linalg.det(A)}")  # ≈ 0

# Solución de mínimos cuadrados (aproximación)
x_aprox = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"Solución aproximada (mínimos cuadrados): {x_aprox}")
```

### 5.4 Eliminación Gaussiana Manual

```python
import numpy as np

def eliminacion_gaussiana(A, b):
    """
    Resuelve Ax = b usando eliminación gaussiana
    """
    n = len(b)
    # Crear matriz aumentada
    Ab = np.column_stack([A.astype(float), b.astype(float)])
    
    # Fase de eliminación
    for i in range(n):
        # Pivoteo parcial (opcional, mejora estabilidad)
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Hacer ceros debajo del pivote
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] -= factor * Ab[i]
    
    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])) / Ab[i, i]
    
    return x

# Ejemplo de uso
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = eliminacion_gaussiana(A, b)
print(f"Solución: {x}")
# [2. 3. -1.]
```

---

## 6. Aplicaciones en Machine Learning

### 6.1 Regresión Lineal

El problema de regresión lineal se formula como:

$$
\min_{\mathbf{w}} \|\mathbf{Xw} - \mathbf{y}\|^2
$$

**Solución analítica (ecuaciones normales):**
$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

Esto es resolver el sistema:
$$
\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}
$$

**Implementación:**
```python
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generar datos de ejemplo
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Agregar columna de unos (bias/intercepto)
X_b = np.c_[np.ones((100, 1)), X]

# Resolver ecuaciones normales: X_b.T @ X_b @ w = X_b.T @ y
w = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

print(f"Intercepto: {w[0]:.2f}")
print(f"Pendiente: {w[1]:.2f}")

# Predicciones
y_pred = X_b @ w

# Visualizar
plt.scatter(X, y, alpha=0.5, label='Datos')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Regresión')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Regresión Lineal usando Sistemas de Ecuaciones')
plt.show()
```

### 6.2 Interpolación de Datos

Encontrar una función que pase por puntos específicos:

**Ejemplo - Interpolación polinómica:**
Dados puntos $(x_1, y_1), (x_2, y_2), (x_3, y_3)$, encontrar $p(x) = a + bx + cx^2$:

$$
\begin{bmatrix}
1 & x_1 & x_1^2 \\
1 & x_2 & x_2^2 \\
1 & x_3 & x_3^2
\end{bmatrix}
\begin{bmatrix} a \\ b \\ c \end{bmatrix}
=
\begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix}
$$

### 6.3 Sistemas de Recomendación

En factorización matricial para recomendaciones:

**Problema:** Completar una matriz de preferencias usuario-ítem
- Usuarios descritos por vectores latentes
- Ítems descritos por vectores latentes
- Resolver sistema para encontrar estos vectores

### 6.4 Computer Vision - Calibración de Cámaras

La proyección de puntos 3D a 2D se modela como:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1 \\ r_{21} & r_{22} & r_{23} & t_2 \\ r_{31} & r_{32} & r_{33} & t_3 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

Calibrar la cámara = resolver sistema lineal para encontrar K y [R|t].

---

## 7. Sistemas Sobredeterminados y Subdeterminados

### 7.1 Sistema Sobredeterminado (m > n)
**Más ecuaciones que incógnitas**

Ejemplo: 3 ecuaciones, 2 incógnitas
$$
\begin{cases}
x + y = 1 \\
2x - y = 3 \\
x + 2y = 2
\end{cases}
$$

Generalmente **no tiene solución exacta**.

**Solución:** Mínimos cuadrados (aproximación)
```python
x_approx = np.linalg.lstsq(A, b, rcond=None)[0]
```

### 7.2 Sistema Subdeterminado (m < n)
**Menos ecuaciones que incógnitas**

Ejemplo: 2 ecuaciones, 3 incógnitas
$$
\begin{cases}
x + y + z = 6 \\
2x - y + 3z = 14
\end{cases}
$$

Generalmente tiene **infinitas soluciones**.

**Solución de norma mínima:**
```python
x_min_norm = np.linalg.lstsq(A, b, rcond=None)[0]
```

---

## 8. Consideraciones Numéricas

### 8.1 Condicionamiento de Matrices

El **número de condición** mide qué tan sensible es la solución a errores:

$$
\text{cond}(A) = \|A\| \cdot \|A^{-1}\|
$$

```python
cond_number = np.linalg.cond(A)

if cond_number > 1e10:
    print("⚠️ Matriz mal condicionada - resultados pueden ser inestables")
elif cond_number > 1e5:
    print("⚠️ Matriz moderadamente mal condicionada")
else:
    print("✅ Matriz bien condicionada")
```

### 8.2 Estabilidad Numérica

**Problemas comunes:**
1. **División por números muy pequeños:** Causa overflow
2. **Cancelación catastrófica:** Restar números casi iguales
3. **Acumulación de errores:** En matrices grandes

**Soluciones:**
- Usar **pivoteo parcial** en eliminación gaussiana
- Preferir **descomposición QR** sobre ecuaciones normales en regresión
- Usar **doble precisión** (float64) cuando sea necesario

---

## 9. Errores Comunes

### ❌ Error 1: Invertir Matrices Singulares
```python
A = np.array([[1, 2], [2, 4]])  # det(A) = 0
# A_inv = np.linalg.inv(A)  # ¡Error!
```

**✅ Solución:** Verificar determinante primero
```python
if abs(np.linalg.det(A)) > 1e-10:
    A_inv = np.linalg.inv(A)
else:
    print("Matriz singular, usar mínimos cuadrados")
```

### ❌ Error 2: Usar Inversa Innecesariamente
```python
# ❌ Ineficiente
x = np.linalg.inv(A) @ b

# ✅ Mejor
x = np.linalg.solve(A, b)  # Más rápido y preciso
```

### ❌ Error 3: No Verificar Dimensiones
```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3
b = np.array([7, 8, 9])  # 3×1
# np.linalg.solve(A, b)  # Error: A debe ser cuadrada
```

---

## 10. Ejercicios Prácticos

### Ejercicio 1: Resolución Básica
Resuelve usando eliminación gaussiana:
$$
\begin{cases}
3x + 2y - z = 1 \\
2x - 2y + 4z = -2 \\
-x + \frac{1}{2}y - z = 0
\end{cases}
$$

### Ejercicio 2: Análisis de Soluciones
Determina sin resolver si cada sistema tiene solución única, infinitas o ninguna:

a) $\begin{cases} x + y = 3 \\ 2x + 2y = 6 \end{cases}$

b) $\begin{cases} x + y = 3 \\ x + y = 5 \end{cases}$

c) $\begin{cases} x + y = 3 \\ x - y = 1 \end{cases}$

### Ejercicio 3: Regresión Lineal Simple
Dados los puntos: (1, 2), (2, 4), (3, 5), (4, 4), (5, 5)
- Encuentra la recta de mejor ajuste $y = mx + b$
- Usa ecuaciones normales
- Grafica los datos y la recta

### Ejercicio 4: Implementación
Implementa la eliminación de Gauss-Jordan desde cero y compara con `np.linalg.solve()`.

---

## 11. Recursos Adicionales

### 📺 Videos
- **3Blue1Brown:** "Inverse matrices, column space and null space"
- **MIT OpenCourseWare:** "Linear Algebra - Lecture 2"

### 📚 Lecturas
- **Introduction to Linear Algebra** (Strang): Capítulos 1-2
- **Numerical Recipes:** Capítulo sobre solución de sistemas lineales

### 🛠️ Herramientas
- **SymPy:** Solución simbólica de sistemas
- **SciPy:** Métodos numéricos avanzados (sparse matrices)

---

## 📌 Resumen Clave

| Aspecto | Detalle |
|---------|---------|
| **Forma matricial** | Ax = b |
| **Solución única** | det(A) ≠ 0 (sistemas cuadrados) |
| **Método principal** | Eliminación gaussiana O(n³) |
| **Python** | `np.linalg.solve(A, b)` |
| **ML** | Regresión lineal, calibración, interpolación |
| **Cuidado** | Matrices mal condicionadas → resultados inestables |

---

## 🎯 Próximos Pasos

**Día 6:** NumPy para Álgebra Lineal
- Operaciones vectorizadas
- Broadcasting
- Optimización de rendimiento
- Aplicaciones prácticas

---

*Los sistemas de ecuaciones lineales son el corazón de muchos algoritmos de ML. ¡Domina este concepto para entender cómo funcionan los modelos!*
