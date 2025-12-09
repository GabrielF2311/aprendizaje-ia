# Día 3: Eigenvalores y Eigenvectores - Fundamentos

## 📋 Objetivos del Día
- Comprender qué son eigenvalores y eigenvectores
- Calcular eigenvalores y eigenvectores manualmente y con NumPy
- Interpretar el significado geométrico
- Reconocer la importancia en Machine Learning
- Diagonalización de matrices

---

## 1. Definición

### 1.1 Conceptos Básicos

Para una matriz cuadrada **A** (n×n), un **eigenvector** $\mathbf{v}$ (vector propio) y su **eigenvalor** $\lambda$ (valor propio) satisfacen:

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

**Interpretación:**
- Cuando A se multiplica por $\mathbf{v}$, el resultado es el mismo vector $\mathbf{v}$ escalado por $\lambda$
- La dirección de $\mathbf{v}$ no cambia, solo su magnitud
- $\lambda$ puede ser positivo, negativo, cero, o incluso complejo

**Condiciones:**
- $\mathbf{v} \neq \mathbf{0}$ (el vector cero no cuenta como eigenvector)
- $\lambda$ puede ser cualquier número (real o complejo)
- Una matriz n×n tiene **n eigenvalores** (contando multiplicidad)

### 1.2 Ejemplo Simple 2×2

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}
$$

**Eigenvalor** $\lambda_1 = 5$:
$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

**Verificación:**
$$
A\mathbf{v}_1 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \lambda_1 \mathbf{v}_1 \quad ✓
$$

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

# Eigenvalor y eigenvector conocidos
lambda1 = 5
v1 = np.array([1, 1])

# Verificar Av = λv
Av = A @ v1
lambda_v = lambda1 * v1

print(f"A @ v = {Av}")
print(f"λ @ v = {lambda_v}")
print(f"¿Iguales? {np.allclose(Av, lambda_v)}")  # True
```

---

## 2. Cálculo de Eigenvalores

### 2.1 Ecuación Característica

De $A\mathbf{v} = \lambda\mathbf{v}$, reorganizamos:

$$
A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}
$$

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

Para que exista solución no trivial ($\mathbf{v} \neq \mathbf{0}$), la matriz $(A - \lambda I)$ debe ser **singular**:

$$
\det(A - \lambda I) = 0
$$

Esta es la **ecuación característica**.

### 2.2 Ejemplo - Matriz 2×2

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}
$$

**Paso 1:** Formar $A - \lambda I$
$$
A - \lambda I = \begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix}
$$

**Paso 2:** Calcular determinante
$$
\det(A - \lambda I) = (4-\lambda)(3-\lambda) - (1)(2)
$$

$$
= 12 - 4\lambda - 3\lambda + \lambda^2 - 2
$$

$$
= \lambda^2 - 7\lambda + 10
$$

**Paso 3:** Resolver $\lambda^2 - 7\lambda + 10 = 0$

Factorizando:
$$
(\lambda - 5)(\lambda - 2) = 0
$$

**Eigenvalores:**
$$
\lambda_1 = 5, \quad \lambda_2 = 2
$$

### 2.3 Ejemplo - Matriz 3×3

$$
A = \begin{bmatrix} 
3 & 1 & 0 \\ 
0 & 3 & 1 \\ 
0 & 0 & 2 
\end{bmatrix}
$$

**Paso 1:** $A - \lambda I$
$$
A - \lambda I = \begin{bmatrix} 
3-\lambda & 1 & 0 \\ 
0 & 3-\lambda & 1 \\ 
0 & 0 & 2-\lambda 
\end{bmatrix}
$$

**Paso 2:** Determinante (matriz triangular superior)
$$
\det(A - \lambda I) = (3-\lambda)(3-\lambda)(2-\lambda)
$$

$$
= (3-\lambda)^2(2-\lambda) = 0
$$

**Eigenvalores:**
$$
\lambda_1 = 3 \quad \text{(multiplicidad 2)}
$$
$$
\lambda_2 = 2 \quad \text{(multiplicidad 1)}
$$

---

## 3. Cálculo de Eigenvectores

Una vez conocidos los eigenvalores, sustituimos en $(A - \lambda I)\mathbf{v} = \mathbf{0}$ para encontrar los eigenvectores.

### 3.1 Ejemplo Completo

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}, \quad \lambda_1 = 5
$$

**Paso 1:** $(A - \lambda_1 I)\mathbf{v} = \mathbf{0}$
$$
\begin{bmatrix} 4-5 & 1 \\ 2 & 3-5 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

$$
\begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

**Paso 2:** Sistema de ecuaciones
$$
\begin{cases}
-v_1 + v_2 = 0 \\
2v_1 - 2v_2 = 0
\end{cases}
$$

Ambas ecuaciones son equivalentes: $v_2 = v_1$

**Paso 3:** Eigenvector (solución general)
$$
\mathbf{v}_1 = t\begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad t \in \mathbb{R}, t \neq 0
$$

Usualmente tomamos $t=1$:
$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

**Para λ₂ = 2:**
$$
\begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

$$
2v_1 + v_2 = 0 \quad \Rightarrow \quad v_2 = -2v_1
$$

$$
\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}
$$

---

## 4. Usando NumPy

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

# Calcular eigenvalores y eigenvectores
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalores:")
print(eigenvalues)  # [5., 2.]

print("\nEigenvectores (columnas):")
print(eigenvectors)
# [[0.707 -0.447]
#  [0.707  0.894]]

# Verificar para cada par (λ, v)
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]  # Columna i
    
    left = A @ v_i
    right = lambda_i * v_i
    
    print(f"\n--- Eigenvalor {i+1}: λ = {lambda_i:.2f} ---")
    print(f"A @ v = {left}")
    print(f"λ @ v = {right}")
    print(f"¿Iguales? {np.allclose(left, right)}")
```

**⚠️ Nota:** NumPy normaliza los eigenvectores (longitud = 1):
$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \quad \rightarrow \quad \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix}
$$

---

## 5. Propiedades

### 5.1 Propiedades Básicas

1. **Traza:** $\sum \lambda_i = \text{tr}(A)$ (suma de elementos diagonales)

2. **Determinante:** $\prod \lambda_i = \det(A)$

3. **Multiplicidad:** Un eigenvalor puede repetirse (multiplicidad algebraica)

4. **Independencia:** Eigenvectores correspondientes a eigenvalores distintos son linealmente independientes

**Verificación:**
```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

eigenvalues = np.linalg.eigvals(A)

# Propiedad 1: Suma de eigenvalores = traza
print(f"Suma de λ: {np.sum(eigenvalues):.2f}")
print(f"Traza de A: {np.trace(A):.2f}")

# Propiedad 2: Producto de eigenvalores = determinante
print(f"\nProducto de λ: {np.prod(eigenvalues):.2f}")
print(f"Determinante de A: {np.linalg.det(A):.2f}")
```

### 5.2 Matrices Especiales

**Matriz Simétrica:**
- Todos los eigenvalores son **reales**
- Eigenvectores son **ortogonales**

```python
# Matriz simétrica
A_sim = np.array([[2, 1],
                  [1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A_sim)

print("Eigenvalores (reales):")
print(eigenvalues)  # [3., 1.]

# Verificar ortogonalidad
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 1]
dot_product = np.dot(v1, v2)
print(f"\nProducto punto v1·v2: {dot_product:.10f}")  # ≈ 0
```

**Matriz Diagonal:**
- Eigenvalores = elementos diagonales
- Eigenvectores = vectores canónicos

```python
D = np.array([[5, 0, 0],
              [0, 3, 0],
              [0, 0, 2]])

eigenvalues = np.linalg.eigvals(D)
print("Eigenvalues de matriz diagonal:")
print(eigenvalues)  # [5., 3., 2.]
```

**Matriz Ortogonal:**
- Eigenvalores tienen magnitud 1: $|\lambda| = 1$

**Matriz Triangular:**
- Eigenvalores = elementos diagonales

---

## 6. Interpretación Geométrica

### 6.1 Visualización 2D

Los eigenvectores representan **direcciones especiales** donde la transformación solo **escala** (no rota).

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1],
              [1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

# Crear grid de puntos
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Aplicar transformación
ellipse = A @ circle

# Graficar
plt.figure(figsize=(12, 5))

# Círculo original
plt.subplot(1, 2, 1)
plt.plot(circle[0], circle[1], 'b-', linewidth=2, label='Círculo original')
plt.arrow(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], 
          head_width=0.1, color='r', label=f'v1 (λ={eigenvalues[0]:.1f})')
plt.arrow(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], 
          head_width=0.1, color='g', label=f'v2 (λ={eigenvalues[1]:.1f})')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Antes de transformación')

# Elipse transformada
plt.subplot(1, 2, 2)
plt.plot(ellipse[0], ellipse[1], 'b-', linewidth=2, label='Elipse transformada')
plt.arrow(0, 0, eigenvalues[0]*eigenvectors[0, 0], eigenvalues[0]*eigenvectors[1, 0], 
          head_width=0.2, color='r', label=f'A@v1 = {eigenvalues[0]:.1f}v1')
plt.arrow(0, 0, eigenvalues[1]*eigenvectors[0, 1], eigenvalues[1]*eigenvectors[1, 1], 
          head_width=0.2, color='g', label=f'A@v2 = {eigenvalues[1]:.1f}v2')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Después de transformación')

plt.tight_layout()
plt.show()
```

**Observación:**
- El círculo se convierte en elipse
- Los ejes de la elipse son los eigenvectores
- Las longitudes son los eigenvalores

### 6.2 Significado Físico

- **λ > 1:** El vector se estira en esa dirección
- **0 < λ < 1:** El vector se contrae
- **λ < 0:** El vector se invierte
- **λ = 0:** El espacio se "colapsa" (matriz singular)

---

## 7. Diagonalización

### 7.1 Concepto

Una matriz A es **diagonalizable** si se puede escribir como:

$$
A = PDP^{-1}
$$

Donde:
- **P:** Matriz de eigenvectores (columnas)
- **D:** Matriz diagonal de eigenvalores
- **P⁻¹:** Inversa de P

**Condición:** A debe tener n eigenvectores linealmente independientes.

### 7.2 Ejemplo

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}
$$

Eigenvalores: λ₁ = 5, λ₂ = 2

Eigenvectores: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$

**Construcción:**
$$
P = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}, \quad
D = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}
$$

**Verificación:**
$$
A = PDP^{-1}
$$

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

# P = matriz de eigenvectores
P = eigenvectors

# D = matriz diagonal de eigenvalores
D = np.diag(eigenvalues)

# Verificar A = PDP^(-1)
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv

print("A original:")
print(A)
print("\nA reconstruida (PDP^-1):")
print(A_reconstructed)
print(f"\n¿Iguales? {np.allclose(A, A_reconstructed)}")
```

### 7.3 Utilidad de la Diagonalización

**Cálculo de potencias:**
$$
A^n = (PDP^{-1})^n = PD^nP^{-1}
$$

Donde $D^n$ es fácil de calcular (elevar cada elemento diagonal a n).

```python
# Calcular A^10 usando diagonalización
n = 10

# Método 1: Directo (lento para n grande)
A_power_direct = np.linalg.matrix_power(A, n)

# Método 2: Diagonalización (más eficiente)
D_power = np.diag(eigenvalues ** n)
A_power_diag = P @ D_power @ P_inv

print(f"A^{n} (directo):")
print(A_power_direct)
print(f"\nA^{n} (diagonalización):")
print(A_power_diag)
print(f"\n¿Iguales? {np.allclose(A_power_direct, A_power_diag)}")
```

---

## 8. Aplicaciones Preliminares

### 8.1 Estabilidad de Sistemas Dinámicos

En sistemas dinámicos $\mathbf{x}_{t+1} = A\mathbf{x}_t$:
- Si todos $|\lambda_i| < 1$, el sistema converge a 0 (estable)
- Si algún $|\lambda_i| > 1$, el sistema diverge (inestable)

```python
# Sistema estable
A_estable = np.array([[0.8, 0.1],
                      [0.1, 0.7]])

eigenvalues_estable = np.linalg.eigvals(A_estable)
print("Eigenvalores (estable):")
print(eigenvalues_estable)
print(f"Magnitudes: {np.abs(eigenvalues_estable)}")  # < 1

# Simular sistema
x = np.array([10, 5])
for t in range(20):
    x = A_estable @ x
    if t % 5 == 0:
        print(f"t={t}: x = {x}")

# x → 0 (sistema estable)
```

### 8.2 Análisis de Varianza (Preámbulo a PCA)

En una matriz de covarianza:
- Eigenvalores = varianza en cada dirección principal
- Eigenvectores = direcciones principales de variación

```python
import numpy as np

# Generar datos 2D correlacionados
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(100)  # Correlación

# Matriz de covarianza
cov_matrix = np.cov(X.T)

# Eigenvalores y eigenvectores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalores (varianzas en direcciones principales):")
print(eigenvalues)

print("\nEigenvectores (direcciones principales):")
print(eigenvectors)

# Visualizar
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)

# Graficar eigenvectores escalados por eigenvalores
for i in range(2):
    plt.arrow(0, 0, 
              eigenvectors[0, i] * np.sqrt(eigenvalues[i]) * 3,
              eigenvectors[1, i] * np.sqrt(eigenvalues[i]) * 3,
              head_width=0.2, color=['r', 'g'][i], linewidth=2,
              label=f'PC{i+1} (λ={eigenvalues[i]:.2f})')

plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Componentes Principales (Eigenvectores de Covarianza)')
plt.show()
```

---

## 9. Ejercicios Prácticos

### Ejercicio 1: Cálculo Manual
Encuentra eigenvalores y eigenvectores de:
$$
A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}
$$

### Ejercicio 2: Verificación
Verifica que para matriz simétrica, los eigenvectores son ortogonales:
$$
A = \begin{bmatrix} 5 & 2 \\ 2 & 5 \end{bmatrix}
$$

### Ejercicio 3: Diagonalización
Diagonaliza la matriz y calcula A⁵:
$$
A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}
$$

### Ejercicio 4: Implementación
Implementa una función que verifique si una matriz es diagonalizable.

---

## 📌 Resumen Clave

| Concepto | Fórmula/Propiedad |
|----------|-------------------|
| **Definición** | $A\mathbf{v} = \lambda\mathbf{v}$ |
| **Ec. característica** | $\det(A - \lambda I) = 0$ |
| **Traza** | $\sum \lambda_i = \text{tr}(A)$ |
| **Determinante** | $\prod \lambda_i = \det(A)$ |
| **Diagonalización** | $A = PDP^{-1}$ |
| **Potencias** | $A^n = PD^nP^{-1}$ |

---

## 🎯 Próximos Pasos

**Día 4:** Eigenvectores - Aplicaciones Avanzadas
- Descomposición espectral
- Aplicaciones en ML
- Algoritmos de cálculo

---

*Los eigenvalores y eigenvectores son el corazón de PCA, SVD, y muchos algoritmos de ML. ¡Domina este concepto!*
