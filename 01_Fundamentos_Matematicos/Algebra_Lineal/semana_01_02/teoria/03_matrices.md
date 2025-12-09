# Día 3: Matrices - Fundamentos

## 📋 Contenido

1. [¿Qué es una matriz?](#qué-es-una-matriz)
2. [Tipos de matrices](#tipos-de-matrices)
3. [Operaciones básicas](#operaciones-básicas)
4. [Propiedades de las matrices](#propiedades-de-las-matrices)
5. [Aplicaciones en IA](#aplicaciones-en-ia)

---

## ¿Qué es una matriz?

Una **matriz** es un arreglo rectangular de números dispuestos en filas y columnas. Es una estructura fundamental en álgebra lineal y se usa extensivamente en inteligencia artificial.

### Notación

Una matriz $A$ de dimensión $m \times n$ (m filas, n columnas):

$$A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

Donde:
- $m$ = número de filas
- $n$ = número de columnas
- $a_{ij}$ = elemento en la fila $i$, columna $j$

### Ejemplo

$$A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}$$

Esta es una matriz $2 \times 3$ (2 filas, 3 columnas).

---

## Tipos de Matrices

### 1. Matriz Cuadrada

Una matriz donde $m = n$ (mismo número de filas y columnas).

$$A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix} \quad (3 \times 3)$$

### 2. Matriz Identidad

Matriz cuadrada con 1s en la diagonal principal y 0s en el resto. Se denota como $I$.

$$I_3 = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}$$

**Propiedad importante**: $A \cdot I = I \cdot A = A$

### 3. Matriz Diagonal

Matriz cuadrada con valores solo en la diagonal principal.

$$D = \begin{bmatrix}
5 & 0 & 0 \\
0 & 3 & 0 \\
0 & 0 & 7
\end{bmatrix}$$

### 4. Matriz Cero (Nula)

Todos sus elementos son cero.

$$O = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}$$

### 5. Matriz Simétrica

Una matriz cuadrada donde $A = A^T$ (es igual a su transpuesta).

$$S = \begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 5 \\
3 & 5 & 6
\end{bmatrix}$$

### 6. Matriz Triangular Superior

Todos los elementos debajo de la diagonal principal son cero.

$$U = \begin{bmatrix}
1 & 2 & 3 \\
0 & 5 & 6 \\
0 & 0 & 9
\end{bmatrix}$$

### 7. Matriz Triangular Inferior

Todos los elementos arriba de la diagonal principal son cero.

$$L = \begin{bmatrix}
1 & 0 & 0 \\
4 & 5 & 0 \\
7 & 8 & 9
\end{bmatrix}$$

---

## Operaciones Básicas

### 1. Suma de Matrices

Solo se pueden sumar matrices de la misma dimensión. Se suman elemento por elemento.

$$A + B = \begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} \\
a_{21} + b_{21} & a_{22} + b_{22}
\end{bmatrix}$$

**Ejemplo**:

$$\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix} + \begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix} = \begin{bmatrix}
6 & 8 \\
10 & 12
\end{bmatrix}$$

### 2. Resta de Matrices

Similar a la suma, elemento por elemento.

$$A - B = \begin{bmatrix}
a_{11} - b_{11} & a_{12} - b_{12} \\
a_{21} - b_{21} & a_{22} - b_{22}
\end{bmatrix}$$

### 3. Multiplicación por Escalar

Multiplica cada elemento de la matriz por un número.

$$k \cdot A = \begin{bmatrix}
k \cdot a_{11} & k \cdot a_{12} \\
k \cdot a_{21} & k \cdot a_{22}
\end{bmatrix}$$

**Ejemplo**:

$$3 \cdot \begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix} = \begin{bmatrix}
3 & 6 \\
9 & 12
\end{bmatrix}$$

### 4. Transposición

La transpuesta de $A$ (denotada $A^T$) intercambia filas por columnas.

Si $A$ es $m \times n$, entonces $A^T$ es $n \times m$.

$$A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix} \quad \Rightarrow \quad A^T = \begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}$$

**Fórmula**: $(A^T)_{ij} = A_{ji}$

---

## Propiedades de las Matrices

### Propiedades de la Suma

1. **Conmutativa**: $A + B = B + A$
2. **Asociativa**: $(A + B) + C = A + (B + C)$
3. **Elemento neutro**: $A + O = A$ (donde $O$ es la matriz cero)
4. **Elemento inverso**: $A + (-A) = O$

### Propiedades de la Multiplicación por Escalar

1. **Distributiva respecto a la suma de matrices**: $k(A + B) = kA + kB$
2. **Distributiva respecto a la suma de escalares**: $(k + l)A = kA + lA$
3. **Asociativa**: $(kl)A = k(lA)$
4. **Identidad**: $1 \cdot A = A$

### Propiedades de la Transposición

1. $(A^T)^T = A$
2. $(A + B)^T = A^T + B^T$
3. $(kA)^T = k(A^T)$
4. $(AB)^T = B^T A^T$ (el orden se invierte)

---

## Aplicaciones en IA

### 1. Representación de Datos

Las matrices son fundamentales para almacenar datasets:

```python
# Cada fila es un ejemplo, cada columna es una característica
datos = [[altura, peso, edad],
         [170, 65, 25],
         [180, 75, 30],
         [165, 60, 22]]
```

En NumPy:
```python
import numpy as np
X = np.array([[170, 65, 25],
              [180, 75, 30],
              [165, 60, 22]])
print(f"Forma: {X.shape}")  # (3, 3)
```

### 2. Imágenes Digitales

Una imagen en escala de grises es una matriz donde cada elemento representa la intensidad del pixel.

```python
# Imagen de 28x28 pixels (como MNIST)
imagen = np.random.randint(0, 256, (28, 28))
print(f"Valor del pixel (0,0): {imagen[0, 0]}")
```

### 3. Redes Neuronales

Los pesos de una capa de red neuronal se almacenan en matrices:

```python
# Matriz de pesos: 784 entradas → 128 neuronas
W = np.random.randn(784, 128)
# Matriz de bias
b = np.zeros((1, 128))
```

### 4. Transformaciones de Datos

Normalización usando operaciones matriciales:

```python
# Centrar los datos (restar la media)
X_centered = X - np.mean(X, axis=0)

# Escalar (dividir por desviación estándar)
X_normalized = X_centered / np.std(X, axis=0)
```

---

## Creación de Matrices en Python

### Con NumPy

```python
import numpy as np

# Matriz desde lista
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Matriz de ceros
zeros = np.zeros((3, 4))  # 3 filas, 4 columnas

# Matriz de unos
ones = np.ones((2, 3))

# Matriz identidad
I = np.eye(4)  # 4x4

# Matriz diagonal
D = np.diag([1, 2, 3, 4])

# Matriz aleatoria (valores entre 0 y 1)
R = np.random.rand(3, 3)

# Matriz aleatoria (distribución normal)
N = np.random.randn(3, 3)

# Matriz con valores en un rango
rangos = np.arange(12).reshape(3, 4)  # 0-11 en 3x4
```

### Acceso a Elementos

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Elemento individual
print(A[0, 0])  # 1 (fila 0, columna 0)
print(A[1, 2])  # 6 (fila 1, columna 2)

# Fila completa
print(A[0, :])  # [1 2 3]

# Columna completa
print(A[:, 1])  # [2 5 8]

# Submatriz
print(A[0:2, 1:3])  # [[2 3]
                     #  [5 6]]
```

---

## Ejercicios Prácticos

### Ejercicio 1: Crear diferentes tipos de matrices

```python
import numpy as np

# 1. Crea una matriz 3x3 con valores del 1 al 9
A = np.arange(1, 10).reshape(3, 3)

# 2. Crea una matriz identidad de 5x5
I = np.eye(5)

# 3. Crea una matriz diagonal con valores [2, 4, 6, 8]
D = np.diag([2, 4, 6, 8])

# 4. Crea una matriz de ceros de 2x5
Z = np.zeros((2, 5))
```

### Ejercicio 2: Operaciones básicas

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Suma
suma = A + B

# Resta
resta = A - B

# Multiplicación por escalar
escalar = 3 * A

# Transposición
A_T = A.T
```

### Ejercicio 3: Representar un dataset

```python
# Dataset de estudiantes: [calificación_mat, calificación_fis, horas_estudio]
estudiantes = np.array([
    [85, 78, 5],
    [90, 88, 7],
    [75, 70, 3],
    [88, 85, 6]
])

print(f"Número de estudiantes: {estudiantes.shape[0]}")
print(f"Número de características: {estudiantes.shape[1]}")
print(f"Calificaciones de matemáticas: {estudiantes[:, 0]}")
print(f"Promedio de horas de estudio: {np.mean(estudiantes[:, 2])}")
```

---

## Resumen

✅ **Conceptos clave del día 3**:

1. Una matriz es un arreglo rectangular de números ($m \times n$)
2. Tipos importantes: identidad, diagonal, simétrica, triangular
3. Operaciones básicas: suma, resta, multiplicación por escalar, transposición
4. Las matrices representan datos, imágenes, pesos de redes neuronales
5. NumPy es la herramienta principal para trabajar con matrices en Python

---

## Recursos Adicionales

- [NumPy Array Creation](https://numpy.org/doc/stable/user/basics.creation.html)
- [Matrix Operations in NumPy](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

---

**Próximo tema**: Multiplicación de Matrices y sus aplicaciones en IA 🚀
