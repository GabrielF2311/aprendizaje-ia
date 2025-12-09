# Día 1: Determinantes

## 📋 Objetivos del Día
- Comprender qué es el determinante de una matriz
- Calcular determinantes usando diferentes métodos
- Interpretar el significado geométrico del determinante
- Aplicar propiedades de determinantes
- Reconocer aplicaciones en Machine Learning

---

## 1. Definición del Determinante

### 1.1 ¿Qué es el Determinante?

El **determinante** es un valor escalar asociado a una matriz cuadrada que proporciona información importante sobre la matriz.

**Notación:**
$$
\det(A) \quad \text{o} \quad |A|
$$

**Propiedades clave:**
- Solo existe para matrices **cuadradas** (n × n)
- Indica si la matriz es **invertible** (det ≠ 0)
- Mide el "factor de escala" de la transformación lineal
- Indica orientación (signo positivo o negativo)

### 1.2 Matrices 2×2

Para una matriz 2×2, el determinante es:

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

$$
\det(A) = ad - bc
$$

**Ejemplo:**
$$
A = \begin{bmatrix} 3 & 1 \\ 2 & 4 \end{bmatrix}
$$

$$
\det(A) = (3)(4) - (1)(2) = 12 - 2 = 10
$$

```python
import numpy as np

A = np.array([[3, 1],
              [2, 4]])

# Método manual
det_manual = A[0,0] * A[1,1] - A[0,1] * A[1,0]
print(f"Determinante manual: {det_manual}")  # 10

# Usando NumPy
det_numpy = np.linalg.det(A)
print(f"Determinante NumPy: {det_numpy}")  # 10.0
```

### 1.3 Matrices 3×3

Para una matriz 3×3, usamos la **regla de Sarrus** o **expansión por cofactores**:

$$
A = \begin{bmatrix} 
a_{11} & a_{12} & a_{13} \\ 
a_{21} & a_{22} & a_{23} \\ 
a_{31} & a_{32} & a_{33} 
\end{bmatrix}
$$

**Regla de Sarrus:**
$$
\det(A) = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32}
$$
$$
- a_{13}a_{22}a_{31} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33}
$$

**Ejemplo:**
$$
A = \begin{bmatrix} 
2 & 1 & 3 \\ 
0 & -1 & 2 \\ 
1 & 4 & -2 
\end{bmatrix}
$$

$$
\begin{align}
\det(A) &= 2(-1)(-2) + 1(2)(1) + 3(0)(4) \\
&\quad - 3(-1)(1) - 2(2)(4) - 1(0)(-2) \\
&= 4 + 2 + 0 + 3 - 16 - 0 \\
&= -7
\end{align}
$$

---

## 2. Métodos de Cálculo

### 2.1 Expansión por Cofactores (Método de Laplace)

Para cualquier matriz n×n, podemos expandir por cualquier fila o columna:

$$
\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}
$$

Donde:
- $M_{ij}$ es el **menor** (determinante de la submatriz sin fila i y columna j)
- $(-1)^{i+j} M_{ij}$ es el **cofactor** $C_{ij}$

**Ejemplo - Expansión por primera fila:**
$$
A = \begin{bmatrix} 
2 & 1 & 3 \\ 
0 & -1 & 2 \\ 
1 & 4 & -2 
\end{bmatrix}
$$

$$
\det(A) = 2 \cdot C_{11} + 1 \cdot C_{12} + 3 \cdot C_{13}
$$

$$
C_{11} = (+1) \begin{vmatrix} -1 & 2 \\ 4 & -2 \end{vmatrix} = (-1)(-2) - (2)(4) = 2 - 8 = -6
$$

$$
C_{12} = (-1) \begin{vmatrix} 0 & 2 \\ 1 & -2 \end{vmatrix} = -(0(-2) - 2(1)) = -(-2) = 2
$$

$$
C_{13} = (+1) \begin{vmatrix} 0 & -1 \\ 1 & 4 \end{vmatrix} = 0(4) - (-1)(1) = 1
$$

$$
\det(A) = 2(-6) + 1(2) + 3(1) = -12 + 2 + 3 = -7
$$

### 2.2 Eliminación Gaussiana

Transformar la matriz a forma triangular usando operaciones elementales:

**Reglas:**
1. Intercambiar filas → multiplica det por -1
2. Multiplicar fila por k → multiplica det por k
3. Sumar múltiplo de una fila a otra → det no cambia

**Ejemplo:**
$$
A = \begin{bmatrix} 
2 & 1 & 3 \\ 
0 & -1 & 2 \\ 
1 & 4 & -2 
\end{bmatrix}
$$

Aplicando eliminación gaussiana (sin cambiar filas):
$$
\rightarrow \begin{bmatrix} 
2 & 1 & 3 \\ 
0 & -1 & 2 \\ 
0 & 3.5 & -3.5 
\end{bmatrix}
\rightarrow \begin{bmatrix} 
2 & 1 & 3 \\ 
0 & -1 & 2 \\ 
0 & 0 & 3.5 
\end{bmatrix}
$$

**Determinante = producto diagonal:**
$$
\det(A) = 2 \times (-1) \times 3.5 = -7
$$

```python
import numpy as np

def det_gaussiana(A):
    """Calcula determinante usando eliminación gaussiana"""
    A = A.astype(float).copy()
    n = len(A)
    det = 1
    
    for i in range(n):
        # Pivoteo (si es necesario)
        if A[i, i] == 0:
            for j in range(i+1, n):
                if A[j, i] != 0:
                    A[[i, j]] = A[[j, i]]
                    det *= -1  # Cambio de signo por intercambio
                    break
        
        # Eliminación
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j] -= factor * A[i]
        
        det *= A[i, i]
    
    return det

A = np.array([[2, 1, 3],
              [0, -1, 2],
              [1, 4, -2]])

print(f"Determinante: {det_gaussiana(A)}")  # -7.0
```

---

## 3. Propiedades de los Determinantes

### 3.1 Propiedades Básicas

1. **Identidad:** $\det(I) = 1$

2. **Transpuesta:** $\det(A^T) = \det(A)$

3. **Producto:** $\det(AB) = \det(A) \cdot \det(B)$

4. **Inversa:** $\det(A^{-1}) = \frac{1}{\det(A)}$

5. **Escalar:** $\det(kA) = k^n \det(A)$ (matriz n×n)

6. **Triangular:** Si A es triangular, $\det(A) = \prod a_{ii}$ (producto diagonal)

**Verificación en Python:**
```python
import numpy as np

A = np.array([[2, 3], [1, 4]])
B = np.array([[1, 2], [3, 1]])

# Propiedad del producto
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_AB = np.linalg.det(A @ B)

print(f"det(A) = {det_A:.2f}")
print(f"det(B) = {det_B:.2f}")
print(f"det(A)·det(B) = {det_A * det_B:.2f}")
print(f"det(AB) = {det_AB:.2f}")
print(f"¿Iguales? {np.isclose(det_A * det_B, det_AB)}")

# Propiedad de la transpuesta
print(f"\ndet(A) = {det_A:.2f}")
print(f"det(A^T) = {np.linalg.det(A.T):.2f}")

# Propiedad escalar (k=2, n=2)
k = 2
det_kA = np.linalg.det(k * A)
print(f"\ndet(2A) = {det_kA:.2f}")
print(f"2² · det(A) = {k**2 * det_A:.2f}")
```

### 3.2 Determinante Cero

**Una matriz tiene determinante 0 si y solo si:**
- Las filas (o columnas) son **linealmente dependientes**
- La matriz NO es invertible (singular)
- El espacio vectorial se "colapsa" a menor dimensión

**Ejemplos:**
```python
# Filas proporcionales
A = np.array([[1, 2],
              [2, 4]])  # Fila 2 = 2 × Fila 1

print(f"det(A) = {np.linalg.det(A)}")  # 0.0

# Fila de ceros
B = np.array([[1, 2, 3],
              [0, 0, 0],
              [4, 5, 6]])

print(f"det(B) = {np.linalg.det(B)}")  # 0.0
```

---

## 4. Interpretación Geométrica

### 4.1 En 2D: Área del Paralelogramo

El valor absoluto del determinante de una matriz 2×2 es el **área del paralelogramo** formado por sus vectores columna.

$$
A = \begin{bmatrix} a & c \\ b & d \end{bmatrix}
$$

Los vectores $\mathbf{v}_1 = [a, b]^T$ y $\mathbf{v}_2 = [c, d]^T$ forman un paralelogramo con área:

$$
\text{Área} = |\det(A)| = |ad - bc|
$$

**Visualización:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Vectores
v1 = np.array([3, 1])
v2 = np.array([1, 2])

A = np.column_stack([v1, v2])
det_A = np.linalg.det(A)

# Graficar paralelogramo
origin = np.array([0, 0])
vertices = np.array([[0, 0],
                     v1,
                     v1 + v2,
                     v2,
                     [0, 0]])

plt.figure(figsize=(8, 6))
plt.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2)
plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)
plt.arrow(0, 0, v1[0], v1[1], head_width=0.2, color='r', label='v1')
plt.arrow(0, 0, v2[0], v2[1], head_width=0.2, color='g', label='v2')
plt.grid(True)
plt.axis('equal')
plt.title(f'Área del paralelogramo = |det(A)| = {abs(det_A):.2f}')
plt.legend()
plt.show()
```

### 4.2 En 3D: Volumen del Paralelepípedo

En 3D, el determinante representa el **volumen del paralelepípedo** formado por tres vectores.

$$
\text{Volumen} = |\det(A)|
$$

Donde A tiene los tres vectores como columnas.

### 4.3 Signo del Determinante

- **det(A) > 0:** La transformación preserva la orientación
- **det(A) < 0:** La transformación invierte la orientación
- **det(A) = 0:** La transformación "aplasta" el espacio a menor dimensión

---

## 5. Aplicaciones

### 5.1 Invertibilidad de Matrices

**Teorema:** Una matriz A es invertible ⟺ det(A) ≠ 0

```python
import numpy as np

def es_invertible(A):
    """Verifica si una matriz es invertible"""
    det = np.linalg.det(A)
    return abs(det) > 1e-10  # Umbral para errores numéricos

# Matriz invertible
A = np.array([[1, 2], [3, 4]])
print(f"A es invertible: {es_invertible(A)}")  # True
print(f"det(A) = {np.linalg.det(A)}")  # -2.0

# Matriz singular
B = np.array([[1, 2], [2, 4]])
print(f"B es invertible: {es_invertible(B)}")  # False
print(f"det(B) = {np.linalg.det(B)}")  # 0.0
```

### 5.2 Regla de Cramer (Sistemas Lineales)

Para sistema Ax = b con det(A) ≠ 0:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

Donde $A_i$ es A con la columna i reemplazada por b.

**Ejemplo:**
$$
\begin{cases}
2x + 3y = 8 \\
x - y = 1
\end{cases}
$$

$$
A = \begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix}, \quad
b = \begin{bmatrix} 8 \\ 1 \end{bmatrix}
$$

$$
x = \frac{\det\begin{bmatrix} 8 & 3 \\ 1 & -1 \end{bmatrix}}{\det(A)} = \frac{-8-3}{-2-3} = \frac{-11}{-5} = 2.2
$$

$$
y = \frac{\det\begin{bmatrix} 2 & 8 \\ 1 & 1 \end{bmatrix}}{\det(A)} = \frac{2-8}{-5} = \frac{-6}{-5} = 1.2
$$

```python
import numpy as np

A = np.array([[2, 3], [1, -1]], dtype=float)
b = np.array([8, 1], dtype=float)

det_A = np.linalg.det(A)

# Calcular x
A_x = A.copy()
A_x[:, 0] = b
x = np.linalg.det(A_x) / det_A

# Calcular y
A_y = A.copy()
A_y[:, 1] = b
y = np.linalg.det(A_y) / det_A

print(f"Solución: x = {x}, y = {y}")  # x = 2.2, y = 1.2

# Verificar
print(f"Verificación: {np.allclose(A @ np.array([x, y]), b)}")
```

⚠️ **Nota:** La regla de Cramer es ineficiente para sistemas grandes. Usar `np.linalg.solve()` en la práctica.

### 5.3 Cálculo de Áreas y Volúmenes

```python
import numpy as np

# Triángulo con vértices (x1,y1), (x2,y2), (x3,y3)
def area_triangulo(p1, p2, p3):
    """Calcula área de triángulo usando determinante"""
    matriz = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])
    return 0.5 * abs(np.linalg.det(matriz))

# Ejemplo
A = [0, 0]
B = [4, 0]
C = [2, 3]

area = area_triangulo(A, B, C)
print(f"Área del triángulo: {area}")  # 6.0
```

### 5.4 En Machine Learning

**1. Detección de Multicolinealidad:**
- En regresión, si det(X^T X) ≈ 0, las características son colineales
- Indica problemas numéricos en el entrenamiento

**2. Análisis de Componentes Principales (PCA):**
- El determinante de la matriz de covarianza indica la varianza total
- det = 0 indica dimensiones redundantes

**3. Redes Neuronales:**
- Jacobiano (matriz de derivadas parciales)
- det(J) mide cómo cambia el volumen bajo la transformación de la red

```python
import numpy as np
from sklearn.datasets import make_regression

# Generar datos con multicolinealidad
X, y = make_regression(n_samples=100, n_features=5, random_state=42)

# Agregar columna redundante (X[:, 0] * 2)
X_colineal = np.column_stack([X, X[:, 0] * 2])

# Verificar determinante de X^T X
XTX = X.T @ X
XTX_colineal = X_colineal.T @ X_colineal

print(f"det(X^T X) sin colinealidad: {np.linalg.det(XTX):.2e}")
print(f"det(X^T X) con colinealidad: {np.linalg.det(XTX_colineal):.2e}")
# Con colinealidad, det ≈ 0
```

---

## 6. Cálculo Eficiente

### 6.1 Complejidad Computacional

| Método | Complejidad |
|--------|-------------|
| Expansión por cofactores | O(n!) - ¡Muy lento! |
| Eliminación gaussiana | O(n³) |
| Algoritmos optimizados (NumPy) | O(n³) con constantes pequeñas |

**Para matrices grandes, siempre usar librerías optimizadas:**

```python
import numpy as np
import time

# Comparación de tiempos
for n in [10, 50, 100, 200]:
    A = np.random.rand(n, n)
    
    start = time.time()
    det = np.linalg.det(A)
    tiempo = time.time() - start
    
    print(f"n={n:3d}: {tiempo:.4f}s")

# Salida ejemplo:
# n= 10: 0.0001s
# n= 50: 0.0005s
# n=100: 0.0020s
# n=200: 0.0120s
```

---

## 7. Errores Comunes

### ❌ Error 1: Matriz No Cuadrada
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

# np.linalg.det(A)  # ¡Error! Solo matrices cuadradas
```

### ❌ Error 2: Asumir det(A+B) = det(A) + det(B)
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# ❌ Incorrecto
print(np.linalg.det(A) + np.linalg.det(B))  # -6.0

# Correcto
print(np.linalg.det(A + B))  # 0.0

# ¡NO son iguales!
```

### ❌ Error 3: Errores Numéricos
```python
# Matriz casi singular
A = np.array([[1, 2],
              [1.0001, 2.0002]])

det = np.linalg.det(A)
print(f"det(A) = {det}")  # Muy cercano a 0, pero no exactamente 0

# Siempre usar umbral
if abs(det) < 1e-10:
    print("Matriz singular (numéricamente)")
```

---

## 8. Ejercicios Prácticos

### Ejercicio 1: Cálculo Manual
Calcula el determinante:
$$
A = \begin{bmatrix} 
3 & 2 & -1 \\ 
1 & 4 & 2 \\ 
2 & -1 & 3 
\end{bmatrix}
$$

### Ejercicio 2: Propiedades
Verifica que det(AB) = det(A)·det(B) para:
$$
A = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 & -1 \\ 2 & 3 \end{bmatrix}
$$

### Ejercicio 3: Área de Polígono
Calcula el área del cuadrilátero con vértices: (0,0), (3,0), (4,2), (1,3)

### Ejercicio 4: Implementación
Implementa la función `det_3x3(A)` que calcule el determinante de una matriz 3×3 sin usar NumPy.

---

## 9. Recursos Adicionales

### 📺 Videos
- **3Blue1Brown:** "The determinant"
- **Khan Academy:** "Determinant of a 3×3 matrix"

### 📚 Lecturas
- **Linear Algebra and Its Applications** (Strang): Capítulo 5
- **Introduction to Linear Algebra** (Strang): Sección sobre determinantes

---

## 📌 Resumen Clave

| Aspecto | Detalle |
|---------|---------|
| **Definición** | Valor escalar de matriz cuadrada |
| **2×2** | ad - bc |
| **Invertibilidad** | det(A) ≠ 0 ⟺ A es invertible |
| **Geométrico** | Área/volumen de transformación |
| **Producto** | det(AB) = det(A)·det(B) |
| **Cálculo** | Eliminación gaussiana O(n³) |

---

## 🎯 Próximos Pasos

**Día 2:** Matriz Inversa
- Cálculo de inversas
- Propiedades
- Aplicaciones en resolución de sistemas

---

*El determinante es una de las herramientas más poderosas del álgebra lineal. ¡Domínalo para entender invertibilidad y transformaciones!*
