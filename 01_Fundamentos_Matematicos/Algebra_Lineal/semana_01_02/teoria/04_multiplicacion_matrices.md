# Día 4: Multiplicación de Matrices

## 📋 Objetivos del Día
- Comprender el proceso de multiplicación de matrices
- Entender las condiciones para que dos matrices se puedan multiplicar
- Aplicar las propiedades de la multiplicación matricial
- Reconocer aplicaciones en Machine Learning

---

## 1. Condiciones para Multiplicar Matrices

### 1.1 Regla Fundamental
Para multiplicar dos matrices **A** y **B**:
- El número de **columnas de A** debe ser igual al número de **filas de B**

$$
A_{m \times n} \cdot B_{n \times p} = C_{m \times p}
$$

**Ejemplo:**
- A es 2×3 (2 filas, 3 columnas)
- B es 3×4 (3 filas, 4 columnas)
- C será 2×4 (2 filas, 4 columnas) ✅

**No válido:**
- A es 2×3
- B es 2×4
- No se pueden multiplicar ❌ (3 ≠ 2)

### 1.2 Dimensiones Resultantes
Si **A** es $m \times n$ y **B** es $n \times p$, entonces **C = AB** es $m \times p$

---

## 2. Proceso de Multiplicación

### 2.1 Definición Matemática
El elemento $c_{ij}$ de la matriz resultado se calcula como:

$$
c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj}
$$

**Interpretación:**
- Toma la fila $i$ de **A**
- Toma la columna $j$ de **B**
- Multiplica elemento por elemento
- Suma todos los productos

### 2.2 Ejemplo Paso a Paso

Multiplicar:
$$
A = \begin{bmatrix} 2 & 3 \\ 1 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 5 & 1 \\ 2 & 3 \end{bmatrix}
$$

**Cálculo de $c_{11}$** (elemento en fila 1, columna 1):
$$
c_{11} = (2 \times 5) + (3 \times 2) = 10 + 6 = 16
$$

**Cálculo de $c_{12}$** (elemento en fila 1, columna 2):
$$
c_{12} = (2 \times 1) + (3 \times 3) = 2 + 9 = 11
$$

**Cálculo de $c_{21}$** (elemento en fila 2, columna 1):
$$
c_{21} = (1 \times 5) + (4 \times 2) = 5 + 8 = 13
$$

**Cálculo de $c_{22}$** (elemento en fila 2, columna 2):
$$
c_{22} = (1 \times 1) + (4 \times 3) = 1 + 12 = 13
$$

**Resultado:**
$$
C = AB = \begin{bmatrix} 16 & 11 \\ 13 & 13 \end{bmatrix}
$$

### 2.3 Visualización del Proceso

```
Fila 1 de A × Columna 1 de B:
[2, 3] • [5, 2]ᵀ = 2×5 + 3×2 = 16

Fila 1 de A × Columna 2 de B:
[2, 3] • [1, 3]ᵀ = 2×1 + 3×3 = 11

Fila 2 de A × Columna 1 de B:
[1, 4] • [5, 2]ᵀ = 1×5 + 4×2 = 13

Fila 2 de A × Columna 2 de B:
[1, 4] • [1, 3]ᵀ = 1×1 + 4×3 = 13
```

---

## 3. Propiedades de la Multiplicación Matricial

### 3.1 NO es Conmutativa
En general: **AB ≠ BA**

**Ejemplo:**
$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

$$
AB = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}, \quad
BA = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}
$$

**¡AB ≠ BA!**

### 3.2 Es Asociativa
**(AB)C = A(BC)**

Puedes agrupar la multiplicación de diferentes formas sin cambiar el resultado.

### 3.3 Es Distributiva
**A(B + C) = AB + AC**

**Ejemplo:**
$$
A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
C = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
$$

$$
A(B + C) = A\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 5 & 4 \\ 3 & 6 \end{bmatrix}
$$

$$
AB + AC = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} + \begin{bmatrix} 3 & 3 \\ 3 & 3 \end{bmatrix} = \begin{bmatrix} 5 & 4 \\ 3 & 6 \end{bmatrix}
$$

### 3.4 Elemento Identidad
**AI = IA = A**

Donde **I** es la matriz identidad.

### 3.5 Transpuesta del Producto
**(AB)ᵀ = BᵀAᵀ**

⚠️ **Nota:** El orden se invierte.

---

## 4. Tipos Especiales de Multiplicación

### 4.1 Multiplicación Matriz-Vector
Una matriz $m \times n$ por un vector $n \times 1$ produce un vector $m \times 1$:

$$
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix} =
\begin{bmatrix} 1×1 + 2×0 + 3×2 \\ 4×1 + 5×0 + 6×2 \end{bmatrix} =
\begin{bmatrix} 7 \\ 16 \end{bmatrix}
$$

### 4.2 Producto Exterior (Outer Product)
Vector columna × Vector fila = Matriz:

$$
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
\begin{bmatrix} 4 & 5 \end{bmatrix} =
\begin{bmatrix} 4 & 5 \\ 8 & 10 \\ 12 & 15 \end{bmatrix}
$$

### 4.3 Producto Hadamard (Element-wise)
Multiplicación elemento por elemento (mismo tamaño):

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \odot
\begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix} =
\begin{bmatrix} 2 & 0 \\ 3 & 8 \end{bmatrix}
$$

⚠️ **Símbolo:** $\odot$ (diferente del producto matricial estándar)

---

## 5. Complejidad Computacional

### 5.1 Algoritmo Básico
Para multiplicar dos matrices $n \times n$:

**Complejidad:** $O(n^3)$

**Operaciones:** $n^3$ multiplicaciones y $n^2(n-1)$ sumas

**Ejemplo:** Para matrices 1000×1000:
- ~1 billón de operaciones
- Tiempo considerable sin optimización

### 5.2 Optimizaciones
1. **Algoritmo de Strassen:** $O(n^{2.807})$
2. **Librerías optimizadas:** NumPy usa BLAS/LAPACK (hasta 100× más rápido)
3. **Hardware especializado:** GPUs para multiplicaciones masivas

---

## 6. Aplicaciones en Machine Learning

### 6.1 Transformación de Datos
```
X (datos)     ×    W (pesos)     =    Y (salida)
[n × d]            [d × m]            [n × m]

n = número de ejemplos
d = dimensión de entrada
m = dimensión de salida
```

**Ejemplo - Capa Dense en Red Neuronal:**
```python
# 100 imágenes de 784 píxeles → 256 características
X: (100, 784)
W: (784, 256)
Y = X @ W → (100, 256)
```

### 6.2 Composición de Transformaciones
En redes neuronales multicapa:

$$
Y = X \cdot W_1 \cdot W_2 \cdot W_3
$$

Cada multiplicación aplica una transformación no lineal (con activaciones).

### 6.3 Batch Processing
Procesar múltiples ejemplos simultáneamente:

$$
\begin{bmatrix}
— \text{ejemplo 1} — \\
— \text{ejemplo 2} — \\
— \text{ejemplo 3} — \\
\vdots
\end{bmatrix}
\times
\begin{bmatrix}
| & | & | \\
w_1 & w_2 & w_3 \\
| & | & |
\end{bmatrix}
=
\begin{bmatrix}
— \text{salida 1} — \\
— \text{salida 2} — \\
— \text{salida 3} — \\
\vdots
\end{bmatrix}
$$

### 6.4 Atención en Transformers
El mecanismo de atención usa multiplicaciones matriciales:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- **Q** (Query): $n \times d_k$
- **K** (Key): $m \times d_k$
- **V** (Value): $m \times d_v$

---

## 7. Implementación en Python

### 7.1 Multiplicación Manual
```python
def multiplicar_matrices(A, B):
    """Multiplicación matricial desde cero"""
    filas_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    
    # Verificar dimensiones
    if cols_A != len(B):
        raise ValueError("Dimensiones incompatibles")
    
    # Inicializar matriz resultado
    C = [[0] * cols_B for _ in range(filas_A)]
    
    # Multiplicación
    for i in range(filas_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

# Ejemplo de uso
A = [[2, 3], [1, 4]]
B = [[5, 1], [2, 3]]
C = multiplicar_matrices(A, B)
print(C)  # [[16, 11], [13, 13]]
```

### 7.2 Con NumPy (Optimizado)
```python
import numpy as np

A = np.array([[2, 3], [1, 4]])
B = np.array([[5, 1], [2, 3]])

# Método 1: Operador @
C = A @ B

# Método 2: np.dot()
C = np.dot(A, B)

# Método 3: np.matmul()
C = np.matmul(A, B)

print(C)
# [[16 11]
#  [13 13]]
```

### 7.3 Comparación de Rendimiento
```python
import numpy as np
import time

n = 1000
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# NumPy optimizado
start = time.time()
C = A @ B
tiempo_numpy = time.time() - start

print(f"NumPy: {tiempo_numpy:.4f} segundos")
# NumPy: ~0.05 segundos (con BLAS)

# Implementación manual sería ~100× más lenta
```

---

## 8. Errores Comunes

### ❌ Error 1: Dimensiones Incompatibles
```python
A = np.array([[1, 2, 3]])      # 1×3
B = np.array([[4, 5], [6, 7]]) # 2×2
# A @ B → Error: 3 ≠ 2
```

**✅ Solución:** Verificar que columnas(A) = filas(B)

### ❌ Error 2: Asumir Conmutatividad
```python
# AB ≠ BA en general
A @ B != B @ A  # Puede dar resultados diferentes
```

### ❌ Error 3: Confundir con Multiplicación Element-wise
```python
# Multiplicación matricial
A @ B  # Producto matricial estándar

# Multiplicación elemento por elemento
A * B  # Producto Hadamard (NumPy)
```

---

## 9. Ejercicios Prácticos

### Ejercicio 1: Multiplicación Básica
Calcula manualmente:
$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}
$$

### Ejercicio 2: Verificar Propiedades
Dadas tres matrices A, B, C de 2×2, verifica:
- (AB)C = A(BC)
- A(B + C) = AB + AC

### Ejercicio 3: Red Neuronal Simple
Implementa la propagación hacia adelante de una capa:
- Entrada: 5 ejemplos de 10 características
- Pesos: 10 neuronas de entrada → 3 de salida
- Calcula la salida

### Ejercicio 4: Optimización
Compara el tiempo de ejecución entre:
- Implementación manual
- NumPy
- Para matrices de tamaño 100, 500, 1000

---

## 10. Recursos Adicionales

### 📺 Videos Recomendados
- **3Blue1Brown:** "Matrix Multiplication as Composition"
- **Khan Academy:** "Matrix Multiplication"

### 📚 Lecturas
- **Deep Learning Book** (Goodfellow): Capítulo 2.2
- **Linear Algebra Done Right** (Axler): Capítulo 3

### 🔧 Herramientas
- **Matrix Multiplication Visualizer:** matrix.reshish.com
- **Wolfram Alpha:** Verificar cálculos

---

## 📌 Resumen Clave

| Concepto | Detalle |
|----------|---------|
| **Condición** | columnas(A) = filas(B) |
| **Resultado** | $A_{m×n} \cdot B_{n×p} = C_{m×p}$ |
| **Conmutativa** | ❌ AB ≠ BA |
| **Asociativa** | ✅ (AB)C = A(BC) |
| **Complejidad** | $O(n^3)$ (algoritmo básico) |
| **ML Principal** | Transformaciones de datos, capas neuronales |

---

## 🎯 Próximos Pasos

**Día 5:** Sistemas de Ecuaciones Lineales
- Representación matricial
- Métodos de solución
- Aplicaciones en ML

---

*Recuerda: La multiplicación de matrices es la operación fundamental en todas las redes neuronales. ¡Practica hasta dominarla!*
