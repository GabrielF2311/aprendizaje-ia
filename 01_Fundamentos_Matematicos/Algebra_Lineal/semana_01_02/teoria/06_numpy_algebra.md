# Día 6: NumPy para Álgebra Lineal

## 📋 Objetivos del Día
- Dominar operaciones de álgebra lineal con NumPy
- Comprender broadcasting y vectorización
- Optimizar código para mejor rendimiento
- Aplicar técnicas eficientes en contextos de Machine Learning
- Conocer diferencias entre operaciones nativas vs. loops tradicionales

---

## 1. Introducción a NumPy

### 1.1 ¿Por Qué NumPy?

**NumPy** (Numerical Python) es la biblioteca fundamental para computación científica en Python.

**Ventajas sobre listas nativas:**
- ⚡ **Velocidad:** 10-100× más rápido (implementado en C)
- 💾 **Memoria:** Usa menos memoria (arrays homogéneos)
- 🔧 **Funcionalidad:** Operaciones vectorizadas integradas
- 🎯 **Sintaxis:** Código más limpio y legible

**Comparación de rendimiento:**
```python
import numpy as np
import time

# Listas Python
n = 1000000
python_list = list(range(n))

start = time.time()
result = [x * 2 for x in python_list]
tiempo_python = time.time() - start

# NumPy
numpy_array = np.arange(n)

start = time.time()
result = numpy_array * 2
tiempo_numpy = time.time() - start

print(f"Python: {tiempo_python:.4f}s")
print(f"NumPy: {tiempo_numpy:.4f}s")
print(f"NumPy es {tiempo_python/tiempo_numpy:.1f}× más rápido")
# NumPy es ~50× más rápido
```

### 1.2 Instalación e Importación

```python
# Instalación (si no está instalado)
# pip install numpy

# Importación estándar
import numpy as np

# Verificar versión
print(np.__version__)  # 1.24.0 o superior
```

---

## 2. Arrays de NumPy

### 2.1 Creación de Arrays

**Desde listas:**
```python
# Vector (1D)
v = np.array([1, 2, 3, 4, 5])
print(v.shape)  # (5,)
print(v.ndim)   # 1 (dimensión)

# Matriz (2D)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)  # (2, 3)
print(M.ndim)   # 2

# Tensor (3D)
T = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(T.shape)  # (2, 2, 2)
print(T.ndim)   # 3
```

**Arrays especiales:**
```python
# Ceros
zeros = np.zeros((3, 4))  # Matriz 3×4 de ceros
print(zeros)

# Unos
ones = np.ones((2, 3))  # Matriz 2×3 de unos

# Identidad
I = np.eye(4)  # Matriz identidad 4×4
print(I)

# Rango
r = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Espaciado lineal
lin = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Aleatorios
rand = np.random.rand(3, 3)  # Valores entre 0 y 1
randn = np.random.randn(3, 3)  # Distribución normal

# Lleno con valor específico
full = np.full((2, 3), 7)  # Matriz 2×3 llena de 7s
```

### 2.2 Atributos Importantes

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(A.shape)      # (3, 4) - dimensiones
print(A.size)       # 12 - total de elementos
print(A.ndim)       # 2 - número de dimensiones
print(A.dtype)      # int64 - tipo de datos
print(A.itemsize)   # 8 - bytes por elemento
print(A.nbytes)     # 96 - bytes totales (12 * 8)
```

### 2.3 Tipos de Datos (dtype)

```python
# Enteros
int32_array = np.array([1, 2, 3], dtype=np.int32)
int64_array = np.array([1, 2, 3], dtype=np.int64)

# Flotantes
float32_array = np.array([1.0, 2.0], dtype=np.float32)
float64_array = np.array([1.0, 2.0], dtype=np.float64)

# Booleanos
bool_array = np.array([True, False, True], dtype=bool)

# Conversión
a = np.array([1, 2, 3])
a_float = a.astype(np.float64)
print(a.dtype, a_float.dtype)  # int64 float64
```

---

## 3. Operaciones Vectorizadas

### 3.1 Operaciones Aritméticas Element-wise

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Suma
print(a + b)  # [11, 22, 33, 44]

# Resta
print(a - b)  # [-9, -18, -27, -36]

# Multiplicación element-wise
print(a * b)  # [10, 40, 90, 160]

# División
print(b / a)  # [10.0, 10.0, 10.0, 10.0]

# Potencia
print(a ** 2)  # [1, 4, 9, 16]

# Operaciones con escalares
print(a + 10)  # [11, 12, 13, 14]
print(a * 2)   # [2, 4, 6, 8]
```

**Comparación con loops:**
```python
# ❌ Forma tradicional (lenta)
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])

# ✅ Forma NumPy (rápida)
result = a + b
```

### 3.2 Funciones Matemáticas Universales (ufuncs)

```python
x = np.array([0, np.pi/2, np.pi])

# Trigonométricas
print(np.sin(x))   # [0.0, 1.0, 0.0]
print(np.cos(x))   # [1.0, 0.0, -1.0]
print(np.tan(x))

# Exponenciales y logarítmicas
print(np.exp(x))   # e^x
print(np.log(x))   # ln(x)
print(np.log10(x)) # log₁₀(x)

# Raíces
print(np.sqrt([1, 4, 9, 16]))  # [1.0, 2.0, 3.0, 4.0]

# Redondeo
print(np.round([1.234, 5.678], 2))  # [1.23, 5.68]
print(np.floor([1.9, 2.1]))         # [1.0, 2.0]
print(np.ceil([1.1, 2.9]))          # [2.0, 3.0]

# Valor absoluto
print(np.abs([-1, -2, 3]))  # [1, 2, 3]
```

### 3.3 Operaciones de Reducción

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Suma
print(np.sum(A))           # 21 (todos los elementos)
print(np.sum(A, axis=0))   # [5, 7, 9] (por columnas)
print(np.sum(A, axis=1))   # [6, 15] (por filas)

# Media
print(np.mean(A))          # 3.5
print(np.mean(A, axis=0))  # [2.5, 3.5, 4.5]

# Mínimo y máximo
print(np.min(A))           # 1
print(np.max(A, axis=1))   # [3, 6]

# Desviación estándar
print(np.std(A))           # ~1.71

# Producto
print(np.prod(A))          # 720 (1*2*3*4*5*6)

# Argmin/Argmax (índice del mín/máx)
print(np.argmax(A))        # 5 (índice plano)
print(np.argmax(A, axis=0)) # [1, 1, 1] (por columnas)
```

---

## 4. Álgebra Lineal con NumPy

### 4.1 Producto Punto (Dot Product)

```python
# Vectores
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Método 1: np.dot()
dot_product = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Método 2: @
dot_product = a @ b  # 32

# Método 3: Método de array
dot_product = a.dot(b)  # 32

print(dot_product)  # 32
```

### 4.2 Multiplicación de Matrices

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Método 1: @ (recomendado en Python 3.5+)
C = A @ B

# Método 2: np.matmul()
C = np.matmul(A, B)

# Método 3: np.dot()
C = np.dot(A, B)

print(C)
# [[19 22]
#  [43 50]]

# ⚠️ NO usar * para multiplicación matricial
print(A * B)  # Multiplicación element-wise (Hadamard)
# [[5  12]
#  [21 32]]
```

### 4.3 Matriz Transpuesta

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Método 1: .T
A_T = A.T

# Método 2: np.transpose()
A_T = np.transpose(A)

print(A_T)
# [[1 4]
#  [2 5]
#  [3 6]]

print(A.shape)    # (2, 3)
print(A_T.shape)  # (3, 2)
```

### 4.4 Matriz Inversa

```python
A = np.array([[1, 2],
              [3, 4]])

# Calcular inversa
A_inv = np.linalg.inv(A)

print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verificar: A @ A_inv = I
I = A @ A_inv
print(np.round(I, 10))  # Redondear errores numéricos
# [[1. 0.]
#  [0. 1.]]

# Verificar que es invertible
det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")  # -2.0 (≠ 0, entonces es invertible)
```

### 4.5 Determinante

```python
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

det = np.linalg.det(A)
print(f"Determinante: {det}")  # 1.0
```

### 4.6 Eigenvalues y Eigenvectors

```python
A = np.array([[4, -2],
              [1,  1]])

# Calcular eigenvalues y eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)  # [3., 2.]

print("\nEigenvectors:")
print(eigenvectors)
# [[0.89442719 0.70710678]
#  [0.4472136  0.70710678]]

# Verificar: A @ v = λ @ v
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    
    left = A @ v_i
    right = lambda_i * v_i
    
    print(f"\nλ_{i+1} = {lambda_i}")
    print(f"A @ v = {left}")
    print(f"λ @ v = {right}")
    print(f"¿Iguales? {np.allclose(left, right)}")
```

### 4.7 Normas

```python
v = np.array([3, 4])

# Norma L2 (euclidiana)
norm_l2 = np.linalg.norm(v)  # √(3² + 4²) = 5.0

# Norma L1 (Manhattan)
norm_l1 = np.linalg.norm(v, ord=1)  # |3| + |4| = 7.0

# Norma infinito
norm_inf = np.linalg.norm(v, ord=np.inf)  # max(|3|, |4|) = 4.0

print(f"L2: {norm_l2}, L1: {norm_l1}, L∞: {norm_inf}")
```

### 4.8 Solución de Sistemas Lineales

```python
# Sistema: 3x + y = 9
#          x + 2y = 8

A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

# Resolver Ax = b
x = np.linalg.solve(A, b)
print(f"Solución: x = {x[0]}, y = {x[1]}")  # x = 2.0, y = 3.0

# Verificar
print(f"Verificación: A @ x = {A @ x}")  # [9., 8.]
```

### 4.9 Descomposiciones

**SVD (Singular Value Decomposition):**
```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# A = U @ Σ @ V^T
U, S, VT = np.linalg.svd(A)

print("U (3×2):")
print(U)
print("\nS (valores singulares):")
print(S)  # [9.508, 0.773]
print("\nV^T (2×2):")
print(VT)

# Reconstruir A
Sigma = np.zeros((3, 2))
Sigma[:2, :2] = np.diag(S)
A_reconstructed = U @ Sigma @ VT
print("\nA reconstruida:")
print(A_reconstructed)
```

**QR Decomposition:**
```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

# A = Q @ R
Q, R = np.linalg.qr(A)

print("Q (ortonormal):")
print(Q)
print("\nR (triangular superior):")
print(R)

# Verificar
print("\nQ @ R:")
print(Q @ R)  # Debería ser igual a A
```

---

## 5. Broadcasting

### 5.1 Concepto

**Broadcasting** permite realizar operaciones entre arrays de diferentes formas sin copiar datos explícitamente.

**Reglas:**
1. Si los arrays tienen diferente número de dimensiones, agregar dimensiones de tamaño 1 al inicio del array más pequeño
2. Arrays son compatibles en una dimensión si tienen el mismo tamaño o uno tiene tamaño 1
3. Después del broadcasting, cada array se comporta como si tuviera la forma del array más grande

### 5.2 Ejemplos Básicos

```python
# Escalar + Array
a = np.array([1, 2, 3])
print(a + 5)  # [6, 7, 8]

# Vector + Matriz (por filas)
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

v = np.array([10, 20, 30])

result = M + v  # v se replica en cada fila
print(result)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]

# Vector columna + Matriz (por columnas)
v_col = np.array([[10],
                  [20],
                  [30]])

result = M + v_col
print(result)
# [[11 12 13]
#  [24 25 26]
#  [37 38 39]]
```

### 5.3 Broadcasting en 2D

```python
# (3, 4) + (1, 4)
A = np.ones((3, 4))
B = np.array([[1, 2, 3, 4]])

C = A + B  # B se replica en las 3 filas
print(C.shape)  # (3, 4)

# (3, 1) + (1, 4) → (3, 4)
A = np.array([[1],
              [2],
              [3]])  # (3, 1)

B = np.array([[10, 20, 30, 40]])  # (1, 4)

C = A + B  # Broadcasting a (3, 4)
print(C)
# [[11 21 31 41]
#  [12 22 32 42]
#  [13 23 33 43]]
```

### 5.4 Ejemplo: Normalización

```python
# Datos: 100 muestras de 4 características
X = np.random.randn(100, 4)

# Calcular media por columna (cada característica)
mean = np.mean(X, axis=0)  # (4,)
print(f"Mean shape: {mean.shape}")

# Calcular desviación estándar por columna
std = np.std(X, axis=0)  # (4,)

# Normalizar (Z-score normalization)
X_normalized = (X - mean) / std  # Broadcasting automático

print(f"Media después de normalizar: {np.mean(X_normalized, axis=0)}")
# Cercano a [0, 0, 0, 0]

print(f"Std después de normalizar: {np.std(X_normalized, axis=0)}")
# Cercano a [1, 1, 1, 1]
```

---

## 6. Indexación y Slicing

### 6.1 Indexación Básica

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

# Elemento individual
print(A[0, 1])  # 2 (fila 0, columna 1)

# Fila completa
print(A[1])  # [5, 6, 7, 8]
print(A[1, :])  # Equivalente

# Columna completa
print(A[:, 2])  # [3, 7, 11]

# Submatriz
print(A[0:2, 1:3])
# [[2 3]
#  [6 7]]
```

### 6.2 Indexación Avanzada

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Indexación booleana
mask = A > 5
print(mask)
# [[False False False]
#  [False False  True]
#  [ True  True  True]]

print(A[mask])  # [6, 7, 8, 9]

# Modificar usando máscara
A[A > 5] = 0
print(A)
# [[1 2 3]
#  [4 5 0]
#  [0 0 0]]

# Indexación fancy (con arrays)
indices = np.array([0, 2])
print(A[indices])  # Filas 0 y 2
# [[1 2 3]
#  [0 0 0]]
```

### 6.3 Modificación In-Place

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Modificar elemento
A[0, 1] = 99
print(A)

# Modificar fila
A[1] = [10, 20, 30]
print(A)

# Modificar con condición
A[A < 20] = 0
print(A)
# [[ 0 99  0]
#  [ 0 20 30]]
```

---

## 7. Reshape y Manipulación de Formas

### 7.1 Reshape

```python
a = np.arange(12)  # [0, 1, 2, ..., 11]

# Convertir a matriz 3×4
b = a.reshape(3, 4)
print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Convertir a matriz 4×3
c = a.reshape(4, 3)

# Usar -1 para inferir dimensión
d = a.reshape(2, -1)  # 2 filas, inferir columnas → (2, 6)
print(d.shape)  # (2, 6)
```

### 7.2 Flatten y Ravel

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Flatten (crea copia)
flat = A.flatten()
print(flat)  # [1, 2, 3, 4, 5, 6]

flat[0] = 999
print(A[0, 0])  # 1 (A no cambió)

# Ravel (vista, más eficiente)
rav = A.ravel()
rav[0] = 999
print(A[0, 0])  # 999 (A cambió)
```

### 7.3 Concatenación y Apilado

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Concatenar verticalmente (apilar filas)
C = np.vstack([A, B])
# Equivalente: np.concatenate([A, B], axis=0)
print(C)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Concatenar horizontalmente (apilar columnas)
D = np.hstack([A, B])
# Equivalente: np.concatenate([A, B], axis=1)
print(D)
# [[1 2 5 6]
#  [3 4 7 8]]

# np.stack (nueva dimensión)
E = np.stack([A, B], axis=0)
print(E.shape)  # (2, 2, 2)
```

---

## 8. Aplicaciones en Machine Learning

### 8.1 Normalización Min-Max

```python
def min_max_normalization(X):
    """Escala datos al rango [0, 1]"""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)

# Ejemplo
X = np.array([[1, 200],
              [2, 300],
              [3, 400],
              [4, 500]])

X_norm = min_max_normalization(X)
print(X_norm)
# [[0.   0.  ]
#  [0.33 0.33]
#  [0.67 0.67]
#  [1.   1.  ]]
```

### 8.2 Distancia Euclidiana (Batch)

```python
def euclidean_distances(X, Y):
    """
    Calcula distancia euclidiana entre cada par de filas en X e Y
    X: (n, d)
    Y: (m, d)
    Retorna: (n, m)
    """
    # Usando broadcasting
    # (n, 1, d) - (1, m, d) = (n, m, d)
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances

# Ejemplo
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

Y = np.array([[0, 0],
              [1, 1]])

distances = euclidean_distances(X, Y)
print(distances)
# [[2.236 1.414]  # Distancias de [1,2] a [0,0] y [1,1]
#  [5.    2.828]  # Distancias de [3,4] a [0,0] y [1,1]
#  [7.810 5.657]] # Distancias de [5,6] a [0,0] y [1,1]
```

### 8.3 Softmax (Estable Numéricamente)

```python
def softmax(x):
    """
    Calcula softmax de forma numéricamente estable
    """
    # Restar el máximo para evitar overflow
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Ejemplo
logits = np.array([[2.0, 1.0, 0.1],
                   [1.0, 3.0, 0.5]])

probs = softmax(logits)
print(probs)
# [[0.659 0.242 0.099]
#  [0.119 0.858 0.073]]

print(np.sum(probs, axis=1))  # [1., 1.] (suma a 1)
```

### 8.4 One-Hot Encoding

```python
def one_hot_encode(labels, num_classes):
    """Convierte etiquetas a formato one-hot"""
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1
    return one_hot

# Ejemplo
labels = np.array([0, 2, 1, 2, 0])
one_hot = one_hot_encode(labels, num_classes=3)
print(one_hot)
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
```

### 8.5 Batch Matrix Multiplication

```python
# Multiplicar múltiples matrices al mismo tiempo
# Útil en redes neuronales

# 32 ejemplos, cada uno es una matriz 4×5
batch_A = np.random.randn(32, 4, 5)

# 32 matrices de pesos 5×3
batch_B = np.random.randn(32, 5, 3)

# Multiplicación batch: cada A[i] @ B[i]
result = batch_A @ batch_B  # (32, 4, 3)

print(result.shape)  # (32, 4, 3)

# Verificar manualmente
for i in range(32):
    manual = batch_A[i] @ batch_B[i]
    assert np.allclose(manual, result[i])
```

---

## 9. Optimización de Rendimiento

### 9.1 Evitar Loops

```python
import time

n = 1000000

# ❌ Con loop (lento)
start = time.time()
result = np.zeros(n)
for i in range(n):
    result[i] = i ** 2
tiempo_loop = time.time() - start

# ✅ Vectorizado (rápido)
start = time.time()
result = np.arange(n) ** 2
tiempo_vectorizado = time.time() - start

print(f"Loop: {tiempo_loop:.4f}s")
print(f"Vectorizado: {tiempo_vectorizado:.4f}s")
print(f"Speedup: {tiempo_loop/tiempo_vectorizado:.1f}×")
# Speedup: ~100×
```

### 9.2 Operaciones In-Place

```python
A = np.random.rand(1000, 1000)

# Sin in-place (crea copia)
B = A + 1  # Nueva matriz

# In-place (más eficiente)
A += 1  # Modifica A directamente

# Otras operaciones in-place
A *= 2
A -= 0.5
```

### 9.3 Usar Funciones NumPy Nativas

```python
# ❌ Lento
result = np.array([np.sin(x) for x in range(1000)])

# ✅ Rápido
result = np.sin(np.arange(1000))
```

### 9.4 Preallocate Arrays

```python
# ❌ Append repetido (muy lento)
result = np.array([])
for i in range(10000):
    result = np.append(result, i)

# ✅ Preallocate (rápido)
result = np.zeros(10000)
for i in range(10000):
    result[i] = i

# ✅✅ Mejor: Vectorizado
result = np.arange(10000)
```

---

## 10. Errores Comunes

### ❌ Error 1: Modificar Vista sin Querer

```python
A = np.array([[1, 2], [3, 4]])
B = A  # B es una vista, no una copia

B[0, 0] = 999
print(A[0, 0])  # 999 (A cambió también)

# ✅ Solución: Copiar explícitamente
B = A.copy()
B[0, 0] = 999
print(A[0, 0])  # 1 (A no cambió)
```

### ❌ Error 2: Confundir * con @

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# * es element-wise (Hadamard)
print(A * B)
# [[ 5 12]
#  [21 32]]

# @ es multiplicación matricial
print(A @ B)
# [[19 22]
#  [43 50]]
```

### ❌ Error 3: Broadcasting Inesperado

```python
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[4], [5]])   # (2, 1)

# Broadcasting crea (2, 3)
C = A + B
print(C.shape)  # (2, 3)
print(C)
# [[5 6 7]
#  [6 7 8]]

# Si esto no es lo esperado, verificar dimensiones
```

---

## 11. Ejercicios Prácticos

### Ejercicio 1: Implementar K-Means (1 iteración)
```python
# Datos: 100 puntos en 2D
X = np.random.randn(100, 2)

# 3 centroides iniciales
centroids = np.random.randn(3, 2)

# TODO:
# 1. Calcular distancias de cada punto a cada centroide
# 2. Asignar cada punto al centroide más cercano
# 3. Actualizar centroides como media de sus puntos
```

### Ejercicio 2: Implementar Gradiente Descendente
```python
# Función: f(x) = x^2
# Derivada: f'(x) = 2x
# TODO: Implementar 100 pasos de gradiente descendente
```

### Ejercicio 3: Matriz de Confusión
```python
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 2, 2])

# TODO: Crear matriz de confusión 3×3 sin loops
```

### Ejercicio 4: Implementar ReLU y su Derivada
```python
def relu(x):
    # TODO: Implementar ReLU sin loops
    pass

def relu_derivative(x):
    # TODO: Implementar derivada de ReLU
    pass
```

---

## 12. Recursos Adicionales

### 📚 Documentación
- **NumPy Official Docs:** https://numpy.org/doc/
- **NumPy User Guide:** Fundamentos y conceptos avanzados

### 📺 Videos
- **NumPy Tutorial - freeCodeCamp**
- **Array Programming with NumPy** (Nature paper)

### 🛠️ Herramientas
- **NumPy Exercises:** https://github.com/rougier/numpy-100
- **Jupyter Notebooks:** Para experimentación interactiva

---

## 📌 Resumen Clave

| Operación | Sintaxis | Nota |
|-----------|----------|------|
| **Dot product** | `a @ b` o `np.dot(a, b)` | Vectores |
| **Matrix multiply** | `A @ B` | Matricial |
| **Element-wise** | `A * B` | Hadamard |
| **Transpose** | `A.T` | Transpuesta |
| **Inverse** | `np.linalg.inv(A)` | A debe ser invertible |
| **Solve Ax=b** | `np.linalg.solve(A, b)` | Más eficiente que inv |
| **Norm** | `np.linalg.norm(v)` | L2 por defecto |

**Regla de oro:** ¡Evita loops! Usa vectorización siempre que sea posible.

---

## 🎯 Próximos Pasos

**Día 7:** Transformaciones Lineales
- Matrices de rotación
- Scaling y traslación
- Aplicaciones en Computer Vision
- Proyecciones

---

*NumPy es el corazón de todo el stack de ciencia de datos en Python. ¡Dominarlo te hará 10× más productivo en Machine Learning!*
