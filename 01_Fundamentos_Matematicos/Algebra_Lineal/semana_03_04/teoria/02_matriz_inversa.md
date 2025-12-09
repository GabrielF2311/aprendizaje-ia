# Día 2: Matriz Inversa

## 📋 Objetivos del Día
- Comprender el concepto de matriz inversa
- Calcular inversas usando diferentes métodos
- Aplicar propiedades de matrices inversas
- Usar inversas para resolver sistemas lineales
- Reconocer cuándo NO usar inversas en Machine Learning

---

## 1. Concepto de Matriz Inversa

### 1.1 Definición

La **matriz inversa** de A (denotada $A^{-1}$) es la matriz que satisface:

$$
A \cdot A^{-1} = A^{-1} \cdot A = I
$$

Donde **I** es la matriz identidad.

**Propiedades:**
- Solo matrices **cuadradas** pueden tener inversa
- No todas las matrices cuadradas tienen inversa
- Si existe, la inversa es **única**

### 1.2 Matrices Invertibles (No Singulares)

Una matriz A es **invertible** si y solo si:
- det(A) ≠ 0
- Las filas/columnas son linealmente independientes
- El rango es máximo (rango(A) = n para matriz n×n)

**Ejemplo - Matriz 2×2:**
$$
A = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix}
$$

$$
A^{-1} = \frac{1}{5} \begin{bmatrix} 4 & -1 \\ -3 & 2 \end{bmatrix} = \begin{bmatrix} 0.8 & -0.2 \\ -0.6 & 0.4 \end{bmatrix}
$$

**Verificación:**
$$
A \cdot A^{-1} = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 0.8 & -0.2 \\ -0.6 & 0.4 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

```python
import numpy as np

A = np.array([[2, 1],
              [3, 4]])

A_inv = np.linalg.inv(A)
print("A^(-1):")
print(A_inv)

# Verificar A @ A^(-1) = I
I = A @ A_inv
print("\nA @ A^(-1):")
print(np.round(I, 10))  # Redondear errores numéricos
# [[1. 0.]
#  [0. 1.]]
```

---

## 2. Métodos de Cálculo

### 2.1 Fórmula para Matriz 2×2

Para matriz 2×2:
$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

$$
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

**Pasos:**
1. Calcular det(A) = ad - bc
2. Intercambiar elementos diagonales (a ↔ d)
3. Cambiar signo de elementos fuera de la diagonal
4. Dividir todo por det(A)

**Ejemplo:**
$$
A = \begin{bmatrix} 3 & 1 \\ 2 & 4 \end{bmatrix}
$$

$$
\det(A) = 3(4) - 1(2) = 10
$$

$$
A^{-1} = \frac{1}{10} \begin{bmatrix} 4 & -1 \\ -2 & 3 \end{bmatrix} = \begin{bmatrix} 0.4 & -0.1 \\ -0.2 & 0.3 \end{bmatrix}
$$

```python
def inversa_2x2(A):
    """Calcula inversa de matriz 2×2"""
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    det = a*d - b*c
    
    if abs(det) < 1e-10:
        raise ValueError("Matriz singular (det ≈ 0)")
    
    return (1/det) * np.array([[d, -b],
                                [-c, a]])

A = np.array([[3, 1],
              [2, 4]], dtype=float)

A_inv = inversa_2x2(A)
print("Inversa calculada:")
print(A_inv)

# Comparar con NumPy
print("\nInversa NumPy:")
print(np.linalg.inv(A))
```

### 2.2 Método de Gauss-Jordan

Transforma [A | I] en [I | A⁻¹] usando operaciones elementales.

**Ejemplo:**
$$
A = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix}
$$

**Paso 1:** Matriz aumentada
$$
\left[\begin{array}{cc|cc}
2 & 1 & 1 & 0 \\
3 & 4 & 0 & 1
\end{array}\right]
$$

**Paso 2:** $F_1 = F_1 / 2$
$$
\left[\begin{array}{cc|cc}
1 & 0.5 & 0.5 & 0 \\
3 & 4 & 0 & 1
\end{array}\right]
$$

**Paso 3:** $F_2 = F_2 - 3F_1$
$$
\left[\begin{array}{cc|cc}
1 & 0.5 & 0.5 & 0 \\
0 & 2.5 & -1.5 & 1
\end{array}\right]
$$

**Paso 4:** $F_2 = F_2 / 2.5$
$$
\left[\begin{array}{cc|cc}
1 & 0.5 & 0.5 & 0 \\
0 & 1 & -0.6 & 0.4
\end{array}\right]
$$

**Paso 5:** $F_1 = F_1 - 0.5F_2$
$$
\left[\begin{array}{cc|cc}
1 & 0 & 0.8 & -0.2 \\
0 & 1 & -0.6 & 0.4
\end{array}\right]
$$

$$
A^{-1} = \begin{bmatrix} 0.8 & -0.2 \\ -0.6 & 0.4 \end{bmatrix}
$$

```python
def inversa_gauss_jordan(A):
    """Calcula inversa usando Gauss-Jordan"""
    n = len(A)
    # Crear matriz aumentada [A | I]
    augmented = np.hstack([A.astype(float), np.eye(n)])
    
    # Eliminación hacia adelante
    for i in range(n):
        # Pivoteo (opcional)
        if augmented[i, i] == 0:
            for j in range(i+1, n):
                if augmented[j, i] != 0:
                    augmented[[i, j]] = augmented[[j, i]]
                    break
        
        # Normalizar fila pivote
        augmented[i] = augmented[i] / augmented[i, i]
        
        # Eliminar debajo y arriba
        for j in range(n):
            if i != j:
                augmented[j] -= augmented[j, i] * augmented[i]
    
    # Extraer A^(-1) (lado derecho)
    return augmented[:, n:]

A = np.array([[2, 1],
              [3, 4]])

A_inv = inversa_gauss_jordan(A)
print("Inversa Gauss-Jordan:")
print(A_inv)
```

### 2.3 Usando Matriz de Cofactores (Adjunta)

Para matriz n×n:

$$
A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
$$

Donde **adj(A)** es la **matriz adjunta** (transpuesta de la matriz de cofactores).

**Ejemplo 3×3:**
$$
A = \begin{bmatrix} 
1 & 2 & 3 \\ 
0 & 1 & 4 \\ 
5 & 6 & 0 
\end{bmatrix}
$$

**Paso 1:** Calcular cofactores
$$
C_{11} = (+1)\begin{vmatrix} 1 & 4 \\ 6 & 0 \end{vmatrix} = -24
$$

$$
C_{12} = (-1)\begin{vmatrix} 0 & 4 \\ 5 & 0 \end{vmatrix} = 20
$$

... (continuar para todos)

**Paso 2:** Matriz de cofactores → Transponer → Dividir por det(A)

⚠️ **Este método es ineficiente para matrices grandes** (solo útil para entender el concepto).

---

## 3. Propiedades de Matrices Inversas

### 3.1 Propiedades Algebraicas

1. **(A⁻¹)⁻¹ = A**

2. **(AB)⁻¹ = B⁻¹A⁻¹** (orden invertido)

3. **(Aᵀ)⁻¹ = (A⁻¹)ᵀ**

4. **det(A⁻¹) = 1/det(A)**

5. **(kA)⁻¹ = (1/k)A⁻¹** (k ≠ 0)

**Verificación:**
```python
import numpy as np

A = np.array([[2, 1], [3, 4]])
B = np.array([[1, 2], [0, 1]])

A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

# Propiedad 1: (A^-1)^-1 = A
print("(A^-1)^-1 = A:")
print(np.allclose(np.linalg.inv(A_inv), A))  # True

# Propiedad 2: (AB)^-1 = B^-1 A^-1
AB_inv = np.linalg.inv(A @ B)
producto = B_inv @ A_inv
print("\n(AB)^-1 = B^-1 A^-1:")
print(np.allclose(AB_inv, producto))  # True

# Propiedad 3: (A^T)^-1 = (A^-1)^T
print("\n(A^T)^-1 = (A^-1)^T:")
print(np.allclose(np.linalg.inv(A.T), A_inv.T))  # True

# Propiedad 4: det(A^-1) = 1/det(A)
det_A = np.linalg.det(A)
det_A_inv = np.linalg.det(A_inv)
print(f"\ndet(A) = {det_A:.4f}")
print(f"det(A^-1) = {det_A_inv:.4f}")
print(f"1/det(A) = {1/det_A:.4f}")
```

### 3.2 Matrices Especiales

**Matriz Ortogonal:**
Si A es ortogonal (AAᵀ = I), entonces:
$$
A^{-1} = A^T
$$

Calcular la inversa es trivial (solo transponer).

```python
# Ejemplo: Matriz de rotación (ortogonal)
theta = np.pi / 4  # 45 grados
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

R_inv = np.linalg.inv(R)
R_T = R.T

print("R^(-1):")
print(R_inv)
print("\nR^T:")
print(R_T)
print(f"\n¿Son iguales? {np.allclose(R_inv, R_T)}")  # True
```

**Matriz Diagonal:**
Si D es diagonal:
$$
D = \begin{bmatrix} 
d_1 & 0 & 0 \\ 
0 & d_2 & 0 \\ 
0 & 0 & d_3 
\end{bmatrix}, \quad
D^{-1} = \begin{bmatrix} 
1/d_1 & 0 & 0 \\ 
0 & 1/d_2 & 0 \\ 
0 & 0 & 1/d_3 
\end{bmatrix}
$$

```python
D = np.diag([2, 3, 4])
D_inv = np.diag([1/2, 1/3, 1/4])

print("D^(-1) calculada:")
print(np.linalg.inv(D))
print("\nD^(-1) directa:")
print(D_inv)
```

---

## 4. Aplicaciones

### 4.1 Resolución de Sistemas Lineales

**Sistema:** Ax = b

**Solución:** x = A⁻¹b (si A es invertible)

```python
import numpy as np

# Sistema: 2x + y = 5
#          3x + 4y = 11

A = np.array([[2, 1],
              [3, 4]])
b = np.array([5, 11])

# Método 1: Usando inversa (❌ menos eficiente)
A_inv = np.linalg.inv(A)
x = A_inv @ b
print(f"Solución (inversa): x = {x}")

# Método 2: np.linalg.solve (✅ más eficiente)
x = np.linalg.solve(A, b)
print(f"Solución (solve): x = {x}")

# Verificación
print(f"A @ x = {A @ x}")
print(f"b = {b}")
```

⚠️ **Importante:** En la práctica, **nunca** uses A⁻¹ para resolver sistemas. `np.linalg.solve()` es más rápido y numéricamente estable.

### 4.2 Transformaciones Inversas

```python
import numpy as np
import matplotlib.pyplot as plt

# Transformación: Rotación de 45 grados
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Punto original
p = np.array([1, 0])

# Aplicar rotación
p_rotado = R @ p

# Aplicar rotación inversa
R_inv = np.linalg.inv(R)
p_recuperado = R_inv @ p_rotado

print(f"Punto original: {p}")
print(f"Punto rotado: {p_rotado}")
print(f"Punto recuperado: {p_recuperado}")
print(f"¿Igual al original? {np.allclose(p, p_recuperado)}")
```

### 4.3 En Machine Learning

**1. Regresión Lineal (Ecuaciones Normales):**
$$
w = (X^T X)^{-1} X^T y
$$

```python
import numpy as np
from sklearn.datasets import make_regression

# Generar datos
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)

# Agregar columna de unos (bias)
X_b = np.c_[np.ones(100), X]

# Calcular pesos usando ecuaciones normales
# w = (X^T X)^(-1) X^T y
XTX = X_b.T @ X_b
XTy = X_b.T @ y
w = np.linalg.inv(XTX) @ XTy

print(f"Pesos: {w}")

# ✅ Mejor forma (sin calcular inversa explícitamente)
w_mejor = np.linalg.solve(XTX, XTy)
print(f"Pesos (solve): {w_mejor}")
print(f"¿Iguales? {np.allclose(w, w_mejor)}")
```

⚠️ **Problema:** Si X tiene características correlacionadas, XᵀX puede ser casi singular → inversa inestable.

**2. Matriz de Precisión (Inversa de Covarianza):**

En estadística multivariada:
$$
\Sigma^{-1} = \text{Matriz de Precisión}
$$

```python
import numpy as np

# Datos multivariados
X = np.random.randn(1000, 3)

# Matriz de covarianza
cov = np.cov(X.T)

# Matriz de precisión (inversa de covarianza)
precision = np.linalg.inv(cov)

print("Covarianza:")
print(cov)
print("\nPrecisión:")
print(precision)
```

**3. Calibración de Cámaras (Computer Vision):**

Recuperar parámetros intrínsecos de la cámara invirtiendo la matriz de proyección.

---

## 5. Cuándo NO Usar Inversas

### 5.1 Matrices Grandes

**Complejidad:**
- Calcular A⁻¹: O(n³)
- Resolver Ax = b con A⁻¹: O(n³) + O(n²) = O(n³)
- Resolver Ax = b directamente: O(n³) pero con mejores constantes

```python
import numpy as np
import time

for n in [100, 500, 1000]:
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    
    # Método 1: Inversa
    start = time.time()
    A_inv = np.linalg.inv(A)
    x1 = A_inv @ b
    tiempo_inv = time.time() - start
    
    # Método 2: Solve
    start = time.time()
    x2 = np.linalg.solve(A, b)
    tiempo_solve = time.time() - start
    
    print(f"n={n}: Inversa={tiempo_inv:.4f}s, Solve={tiempo_solve:.4f}s")
    print(f"  Speedup: {tiempo_inv/tiempo_solve:.2f}×")

# Salida ejemplo:
# n=100: Inversa=0.0015s, Solve=0.0008s - Speedup: 1.9×
# n=500: Inversa=0.0420s, Solve=0.0180s - Speedup: 2.3×
# n=1000: Inversa=0.1800s, Solve=0.0650s - Speedup: 2.8×
```

### 5.2 Matrices Mal Condicionadas

**Número de condición:**
$$
\kappa(A) = \|A\| \cdot \|A^{-1}\|
$$

Si κ(A) es grande, pequeños errores en A causan grandes errores en A⁻¹.

```python
import numpy as np

# Matriz bien condicionada
A_buena = np.array([[2, 1],
                     [1, 2]])

cond_buena = np.linalg.cond(A_buena)
print(f"Número de condición (buena): {cond_buena:.2f}")  # ~3

# Matriz mal condicionada
A_mala = np.array([[1, 1],
                    [1, 1.0001]])

cond_mala = np.linalg.cond(A_mala)
print(f"Número de condición (mala): {cond_mala:.2e}")  # ~20000

# Invertir matriz mal condicionada es peligroso
A_mala_inv = np.linalg.inv(A_mala)
print("\nInversa de matriz mal condicionada:")
print(A_mala_inv)

# Verificar A @ A^-1 = I
producto = A_mala @ A_mala_inv
print("\nA @ A^(-1) (debería ser I):")
print(producto)
# Puede tener errores numéricos significativos
```

### 5.3 Alternativas Mejores

**Para resolver Ax = b:**
- ✅ `np.linalg.solve(A, b)` - Eliminación gaussiana
- ✅ Descomposición LU, QR, Cholesky (según el caso)

**Para regresión lineal:**
- ✅ `np.linalg.lstsq(X, y)` - Mínimos cuadrados (maneja matrices rectangulares)
- ✅ Regularización (Ridge, Lasso) - Evita problemas de ill-conditioning

---

## 6. Pseudo-Inversa (Moore-Penrose)

Para matrices **no cuadradas** o **singulares**, existe la **pseudo-inversa** A⁺:

$$
A^+ = (A^T A)^{-1} A^T \quad \text{(si } A \text{ tiene rango completo)}
$$

**Propiedades:**
- AA⁺A = A
- A⁺AA⁺ = A⁺
- Solución de mínimos cuadrados: x = A⁺b

```python
import numpy as np

# Matriz no cuadrada (más filas que columnas)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])  # 3×2

# Pseudo-inversa
A_pinv = np.linalg.pinv(A)

print(f"A shape: {A.shape}")
print(f"A^+ shape: {A_pinv.shape}")  # (2, 3)

# Verificar propiedades
print("\nA @ A^+ @ A:")
print(A @ A_pinv @ A)
print("\n¿Igual a A?")
print(np.allclose(A @ A_pinv @ A, A))  # True

# Usar para resolver sistema sobredeterminado
b = np.array([1, 2, 3])
x = A_pinv @ b
print(f"\nSolución de mínimos cuadrados: {x}")
```

---

## 7. Errores Comunes

### ❌ Error 1: Invertir Matriz Singular
```python
A = np.array([[1, 2],
              [2, 4]])  # Filas proporcionales, det = 0

# np.linalg.inv(A)  # ¡Error! LinAlgError: Singular matrix

# Verificar antes de invertir
if abs(np.linalg.det(A)) > 1e-10:
    A_inv = np.linalg.inv(A)
else:
    print("Matriz singular, usar pseudo-inversa")
    A_inv = np.linalg.pinv(A)
```

### ❌ Error 2: Usar Inversa en Vez de Solve
```python
A = np.random.rand(1000, 1000)
b = np.random.rand(1000)

# ❌ Ineficiente y menos preciso
x = np.linalg.inv(A) @ b

# ✅ Correcto
x = np.linalg.solve(A, b)
```

### ❌ Error 3: Asumir (A+B)⁻¹ = A⁻¹ + B⁻¹
```python
A = np.array([[2, 1], [1, 2]])
B = np.array([[1, 0], [0, 1]])

# (A+B)^(-1) ≠ A^(-1) + B^(-1)
print("(A+B)^(-1):")
print(np.linalg.inv(A + B))

print("\nA^(-1) + B^(-1):")
print(np.linalg.inv(A) + np.linalg.inv(B))

# ¡No son iguales!
```

---

## 8. Ejercicios Prácticos

### Ejercicio 1: Cálculo Manual
Calcula la inversa de:
$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 7 \end{bmatrix}
$$

### Ejercicio 2: Gauss-Jordan
Usa el método de Gauss-Jordan para encontrar la inversa de:
$$
A = \begin{bmatrix} 
2 & 1 & 0 \\ 
0 & 3 & 1 \\ 
1 & 0 & 2 
\end{bmatrix}
$$

### Ejercicio 3: Propiedades
Verifica que (AB)⁻¹ = B⁻¹A⁻¹ para matrices 3×3 aleatorias.

### Ejercicio 4: Pseudo-Inversa
Calcula la pseudo-inversa de una matriz 4×2 y verifica que AA⁺A = A.

---

## 9. Recursos Adicionales

### 📺 Videos
- **3Blue1Brown:** "Inverse matrices, column space and null space"
- **Khan Academy:** "Invertible matrices"

### 📚 Lecturas
- **Gilbert Strang:** "Introduction to Linear Algebra" - Capítulo 2

---

## 📌 Resumen Clave

| Aspecto | Detalle |
|---------|---------|
| **Definición** | AA⁻¹ = A⁻¹A = I |
| **Condición** | det(A) ≠ 0 |
| **2×2** | Fórmula directa |
| **n×n** | Gauss-Jordan O(n³) |
| **⚠️ En ML** | NO usar para sistemas, preferir `solve()` |
| **Alternativa** | Pseudo-inversa para matrices no cuadradas |

---

## 🎯 Próximos Pasos

**Día 3:** Eigenvalores y Eigenvectores
- Definición y cálculo
- Interpretación geométrica
- Aplicaciones en PCA

---

*La matriz inversa es fundamental en teoría, pero en la práctica de ML, casi nunca debes calcularla explícitamente. ¡Usa métodos numéricos más estables!*
