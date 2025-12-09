# Día 5: Descomposición en Valores Singulares (SVD)

## 📋 Objetivos del Día
- Comprender qué es SVD y por qué es fundamental en ML
- Calcular la descomposición SVD de una matriz
- Interpretar los valores singulares y vectores singulares
- Aplicar SVD para compresión de datos y reducción de dimensionalidad
- Reconocer aplicaciones en sistemas de recomendación, NLP y visión computacional

---

## 1. Introducción a SVD

### 1.1 ¿Qué es SVD?

**Singular Value Decomposition (SVD)** es una factorización de cualquier matriz $A_{m \times n}$ en tres matrices:

$$
A = U \Sigma V^T
$$

**Componentes:**
- **U** $(m \times m)$: Matriz ortogonal de **vectores singulares izquierdos**
- **Σ** $(m \times n)$: Matriz diagonal con **valores singulares** (σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0)
- **V** $(n \times n)$: Matriz ortogonal de **vectores singulares derechos**

### 1.2 Propiedades Clave

✅ **Funciona con CUALQUIER matriz**
- No necesita ser cuadrada
- No necesita ser invertible
- Siempre existe para cualquier matriz real

✅ **Generalización de eigendecomposición**
- Para matrices simétricas: SVD = Eigendecomposición
- Para matrices no cuadradas: SVD es la única opción

✅ **Interpretación geométrica**
- **V**: Rotación en el espacio de entrada
- **Σ**: Escalamiento en direcciones principales
- **U**: Rotación en el espacio de salida

### 1.3 Comparación: SVD vs Eigendecomposición

| Aspecto | Eigendecomposición | SVD |
|---------|-------------------|-----|
| **Aplicable a** | Solo matrices cuadradas | Cualquier matriz |
| **Forma** | $A = Q\Lambda Q^{-1}$ | $A = U\Sigma V^T$ |
| **Ortogonalidad** | Solo si A es simétrica | Siempre (U y V) |
| **Valores** | Eigenvalores (pueden ser complejos) | Valores singulares (siempre reales ≥ 0) |
| **Estabilidad numérica** | Puede ser inestable | Muy estable |

---

## 2. Cálculo de SVD

### 2.1 Relación con Eigenvalores

Los valores singulares de **A** son las **raíces cuadradas de los eigenvalores** de $A^T A$:

$$
A^T A = (U\Sigma V^T)^T (U\Sigma V^T) = V\Sigma^T U^T U\Sigma V^T = V\Sigma^2 V^T
$$

**Proceso manual:**
1. Calcular $A^T A$ (matriz $n \times n$)
2. Encontrar eigenvalores de $A^T A$ → son $\sigma_i^2$
3. Valores singulares: $\sigma_i = \sqrt{\lambda_i}$
4. Eigenvectores de $A^T A$ forman las columnas de **V**
5. Columnas de **U** se obtienen de: $u_i = \frac{1}{\sigma_i}Av_i$

### 2.2 Ejemplo Paso a Paso (Matriz 3×2)

**Matriz:**
$$
A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \\ 1 & 1 \end{bmatrix}
$$

**Paso 1:** Calcular $A^T A$
$$
A^T A = \begin{bmatrix} 3 & 1 & 1 \\ 1 & 3 & 1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 1 & 3 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 11 & 7 \\ 7 & 11 \end{bmatrix}
$$

**Paso 2:** Eigenvalores de $A^T A$
$$
\det(A^T A - \lambda I) = 0 \Rightarrow \lambda_1 = 18, \quad \lambda_2 = 4
$$

**Paso 3:** Valores singulares
$$
\sigma_1 = \sqrt{18} = 4.24, \quad \sigma_2 = \sqrt{4} = 2.0
$$

**Paso 4:** Eigenvectores de $A^T A$ (columnas de V)
$$
V = \begin{bmatrix} 0.707 & 0.707 \\ 0.707 & -0.707 \end{bmatrix}
$$

**Paso 5:** Calcular U usando $u_i = \frac{1}{\sigma_i}Av_i$

### 2.3 Implementación con NumPy

```python
import numpy as np

# Matriz de ejemplo
A = np.array([[3, 1],
              [1, 3],
              [1, 1]], dtype=float)

# Calcular SVD
U, S, VT = np.linalg.svd(A, full_matrices=True)

print("Matriz U (3×3):")
print(U)
print(f"\nValores singulares: {S}")
print("\nMatriz V^T (2×2):")
print(VT)

# Verificar reconstrucción
Sigma = np.zeros((3, 2))
Sigma[:2, :2] = np.diag(S)
A_reconstructed = U @ Sigma @ VT

print("\nA original:")
print(A)
print("\nA reconstruida (U @ Σ @ V^T):")
print(A_reconstructed)
print(f"\n¿Iguales? {np.allclose(A, A_reconstructed)}")
```

**Salida:**
```
Valores singulares: [4.24264069 2.        ]

¿Iguales? True
```

---

## 3. Interpretación Geométrica

### 3.1 Visualización 2D

Para una matriz 2×2, SVD descompone la transformación en 3 pasos:

```
Entrada → V^T (rotación) → Σ (escalado) → U (rotación) → Salida
```

**Ejemplo visual:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Matriz de transformación
A = np.array([[3, 1],
              [1, 2]])

# SVD
U, S, VT = np.linalg.svd(A)

# Círculo unitario
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Aplicar transformaciones paso a paso
step1 = VT @ circle          # Rotación inicial
step2 = np.diag(S) @ step1   # Escalado
step3 = U @ step2            # Rotación final

# Aplicar A directamente
direct = A @ circle

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Círculo original
axes[0, 0].plot(circle[0], circle[1])
axes[0, 0].set_title('1. Círculo original')
axes[0, 0].axis('equal')
axes[0, 0].grid()

# Paso 1: V^T
axes[0, 1].plot(step1[0], step1[1])
axes[0, 1].set_title('2. Después de V^T (rotación)')
axes[0, 1].axis('equal')
axes[0, 1].grid()

# Paso 2: Σ
axes[0, 2].plot(step2[0], step2[1])
axes[0, 2].set_title('3. Después de Σ (escalado)')
axes[0, 2].axis('equal')
axes[0, 2].grid()

# Paso 3: U
axes[1, 0].plot(step3[0], step3[1])
axes[1, 0].set_title('4. Después de U (rotación final)')
axes[1, 0].axis('equal')
axes[1, 0].grid()

# Comparación
axes[1, 1].plot(direct[0], direct[1], label='A directo', alpha=0.7)
axes[1, 1].plot(step3[0], step3[1], '--', label='SVD reconstruido')
axes[1, 1].set_title('5. Verificación: A vs SVD')
axes[1, 1].legend()
axes[1, 1].axis('equal')
axes[1, 1].grid()

# Elipse resultante
axes[1, 2].plot(circle[0], circle[1], 'b-', alpha=0.3, label='Original')
axes[1, 2].plot(direct[0], direct[1], 'r-', linewidth=2, label='Transformado')
axes[1, 2].set_title('6. Original vs Transformado')
axes[1, 2].legend()
axes[1, 2].axis('equal')
axes[1, 2].grid()

plt.tight_layout()
plt.show()
```

### 3.2 Significado de los Valores Singulares

Los valores singulares $\sigma_i$ representan:
- **Magnitud:** Qué tanto se estira la matriz en cada dirección principal
- **Importancia:** Valores grandes = direcciones importantes
- **Rango:** Número de valores singulares no nulos = rango de la matriz

**Ejemplo:**
```python
A = np.array([[3, 1, 0],
              [1, 3, 0],
              [0, 0, 0]])

U, S, VT = np.linalg.svd(A)
print(f"Valores singulares: {S}")
# [4.24264069 1.41421356 0.        ]

print(f"Rango de A: {np.linalg.matrix_rank(A)}")  # 2
print(f"Valores singulares no nulos: {np.sum(S > 1e-10)}")  # 2
```

---

## 4. SVD Truncado (Aproximación de Bajo Rango)

### 4.1 Concepto

Podemos aproximar **A** usando solo los **k** valores singulares más grandes:

$$
A_k = U_k \Sigma_k V_k^T
$$

Donde:
- $U_k$: Primeras k columnas de U
- $\Sigma_k$: k×k diagonal con los k valores singulares más grandes
- $V_k^T$: Primeras k filas de V^T

**Ventaja:** Reduce dimensionalidad mientras preserva la mayor parte de la información.

### 4.2 Error de Aproximación

El **teorema de Eckart-Young-Mirsky** garantiza que $A_k$ es la mejor aproximación de rango k de A:

$$
\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}
$$

Donde $\|\cdot\|_F$ es la norma de Frobenius.

### 4.3 Implementación: Compresión de Imágenes

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Cargar imagen en escala de grises
img = np.array(Image.open('imagen.jpg').convert('L'))
print(f"Tamaño original: {img.shape}")  # (altura, ancho)

# SVD
U, S, VT = np.linalg.svd(img, full_matrices=False)

# Probar diferentes rangos
ranks = [5, 10, 20, 50, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for idx, k in enumerate(ranks):
    # Reconstruir con k componentes
    A_k = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
    
    # Calcular compresión
    original_size = img.size
    compressed_size = U[:, :k].size + k + VT[:k, :].size
    compression_ratio = original_size / compressed_size
    
    # Error
    error = np.linalg.norm(img - A_k, 'fro') / np.linalg.norm(img, 'fro')
    
    # Visualizar
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(A_k, cmap='gray')
    axes[row, col].set_title(f'Rango {k}\nCompresión: {compression_ratio:.1f}x\nError: {error:.2%}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Energía capturada
def energy_captured(S, k):
    """Porcentaje de información capturada con k componentes"""
    return np.sum(S[:k]**2) / np.sum(S**2)

print("\nEnergía capturada:")
for k in ranks:
    print(f"k={k:3d}: {energy_captured(S, k):.2%}")
```

**Resultados típicos:**
```
Energía capturada:
k=  5: 45.23%
k= 10: 67.89%
k= 20: 85.12%
k= 50: 95.67%
k=100: 98.94%
```

---

## 5. Aplicaciones en Machine Learning

### 5.1 Reducción de Dimensionalidad

SVD es la base de **PCA** (Principal Component Analysis):

```python
def pca_svd(X, n_components):
    """
    PCA usando SVD
    X: (n_samples, n_features)
    """
    # Centrar los datos
    X_centered = X - np.mean(X, axis=0)
    
    # SVD
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    
    # Componentes principales
    components = VT[:n_components]
    
    # Proyectar datos
    X_transformed = X_centered @ components.T
    
    # Varianza explicada
    explained_variance_ratio = (S[:n_components]**2) / np.sum(S**2)
    
    return X_transformed, components, explained_variance_ratio

# Ejemplo con datos sintéticos
np.random.seed(42)
X = np.random.randn(100, 10)  # 100 muestras, 10 características

# Reducir a 2 componentes
X_2d, components, variance_ratio = pca_svd(X, n_components=2)

print(f"Forma original: {X.shape}")
print(f"Forma reducida: {X_2d.shape}")
print(f"Varianza explicada: {variance_ratio}")
# [0.123, 0.098] → primeras 2 componentes explican ~22% de varianza
```

### 5.2 Sistemas de Recomendación (Matrix Factorization)

SVD se usa para completar matrices de calificaciones usuario-ítem:

```python
def recommend_svd(ratings_matrix, k=20):
    """
    ratings_matrix: (n_users, n_items)
    Valores faltantes = 0
    """
    # SVD truncado
    U, S, VT = np.linalg.svd(ratings_matrix, full_matrices=False)
    
    # Mantener solo k componentes
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    
    # Reconstruir matriz
    predicted_ratings = U_k @ S_k @ VT_k
    
    return predicted_ratings

# Ejemplo: 5 usuarios, 4 películas
# 0 = no calificado
ratings = np.array([[5, 3, 0, 1],
                    [4, 0, 0, 1],
                    [1, 1, 0, 5],
                    [1, 0, 0, 4],
                    [0, 1, 5, 4]])

predicted = recommend_svd(ratings, k=2)

print("Ratings originales:")
print(ratings)
print("\nPredicciones SVD:")
print(np.round(predicted, 2))
```

### 5.3 Procesamiento de Lenguaje Natural (LSA)

**Latent Semantic Analysis** usa SVD para encontrar temas en documentos:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Documentos de ejemplo
documents = [
    "machine learning is great",
    "deep learning neural networks",
    "machine learning algorithms",
    "deep neural networks learning",
    "supervised learning classification"
]

# Crear matriz término-documento (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

print(f"Matriz término-documento: {X.shape}")
# (5 documentos, n_términos únicos)

# SVD para LSA
U, S, VT = np.linalg.svd(X, full_matrices=False)

# Reducir a 2 temas latentes
n_topics = 2
U_topics = U[:, :n_topics]
S_topics = np.diag(S[:n_topics])
VT_topics = VT[:n_topics, :]

# Documentos en espacio de temas
docs_in_topic_space = U_topics @ S_topics

print("\nDocumentos representados en 2 temas:")
print(docs_in_topic_space)

# Términos más importantes por tema
feature_names = vectorizer.get_feature_names_out()
for topic_idx in range(n_topics):
    top_indices = np.argsort(np.abs(VT_topics[topic_idx]))[-3:]
    top_terms = [feature_names[i] for i in top_indices]
    print(f"\nTema {topic_idx + 1}: {', '.join(top_terms)}")
```

### 5.4 Denoising (Eliminación de Ruido)

SVD puede limpiar datos ruidosos manteniendo componentes principales:

```python
import numpy as np
import matplotlib.pyplot as plt

# Señal limpia
t = np.linspace(0, 1, 500)
clean_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

# Agregar ruido
noise = 0.5 * np.random.randn(500)
noisy_signal = clean_signal + noise

# Crear matriz de Hankel (para series temporales)
def hankel_matrix(signal, window_size):
    n = len(signal) - window_size + 1
    H = np.zeros((window_size, n))
    for i in range(n):
        H[:, i] = signal[i:i+window_size]
    return H

# SVD para denoising
window = 50
H = hankel_matrix(noisy_signal, window)
U, S, VT = np.linalg.svd(H, full_matrices=False)

# Mantener solo primeras k componentes (señal)
k = 5
H_denoised = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# Reconstruir señal
denoised_signal = np.mean(H_denoised, axis=0)

# Visualizar
plt.figure(figsize=(12, 6))
plt.plot(t[:len(denoised_signal)], clean_signal[:len(denoised_signal)], 
         label='Señal limpia', linewidth=2)
plt.plot(t, noisy_signal, alpha=0.5, label='Señal ruidosa')
plt.plot(t[:len(denoised_signal)], denoised_signal, 
         '--', label=f'SVD denoised (k={k})', linewidth=2)
plt.legend()
plt.title('Denoising con SVD')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid()
plt.show()
```

---

## 6. SVD en Deep Learning

### 6.1 Compresión de Modelos

Reducir parámetros en redes neuronales grandes:

```python
import torch
import torch.nn as nn

# Capa densa original: 1000 → 1000 (1M parámetros)
original_layer = nn.Linear(1000, 1000, bias=False)
W = original_layer.weight.data.numpy()

print(f"Parámetros originales: {W.size:,}")  # 1,000,000

# SVD para factorizar
U, S, VT = np.linalg.svd(W, full_matrices=False)

# Aproximar con rango k
k = 100
U_k = U[:, :k]
S_k = S[:k]
VT_k = VT[:k, :]

# Crear dos capas más pequeñas
layer1 = nn.Linear(1000, k, bias=False)
layer2 = nn.Linear(k, 1000, bias=False)

layer1.weight.data = torch.from_numpy(np.diag(S_k) @ VT_k).float()
layer2.weight.data = torch.from_numpy(U_k).float()

# Parámetros comprimidos
compressed_params = k * 1000 + k * 1000
print(f"Parámetros comprimidos: {compressed_params:,}")  # 200,000
print(f"Compresión: {W.size / compressed_params:.1f}x")  # 5x

# Verificar aproximación
x = torch.randn(1, 1000)
original_output = original_layer(x)
compressed_output = layer2(layer1(x))

error = torch.norm(original_output - compressed_output) / torch.norm(original_output)
print(f"Error de aproximación: {error.item():.2%}")
```

### 6.2 Inicialización de Pesos

SVD se usa en técnicas de inicialización como **SVD initialization**:

```python
def svd_init(shape):
    """Inicialización usando SVD para matrices ortogonales"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

# Usar en PyTorch
weight = torch.from_numpy(svd_init((512, 256))).float()
```

---

## 7. Propiedades Matemáticas

### 7.1 Normas y SVD

**Norma 2 (espectral):**
$$
\|A\|_2 = \sigma_1 \quad \text{(valor singular más grande)}
$$

**Norma de Frobenius:**
$$
\|A\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2}
$$

**Número de condición:**
$$
\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

```python
A = np.array([[4, 0],
              [3, -5]])

U, S, VT = np.linalg.svd(A)

print(f"Valores singulares: {S}")
print(f"Norma 2: {np.max(S)}")
print(f"Norma 2 (NumPy): {np.linalg.norm(A, 2)}")
print(f"Norma Frobenius: {np.sqrt(np.sum(S**2))}")
print(f"Norma F (NumPy): {np.linalg.norm(A, 'fro')}")
print(f"Número de condición: {np.max(S) / np.min(S)}")
```

### 7.2 Pseudoinversa de Moore-Penrose

Para matrices no cuadradas, la **pseudoinversa** se calcula con SVD:

$$
A^+ = V\Sigma^+ U^T
$$

Donde $\Sigma^+$ invierte los valores singulares no nulos:

$$
\Sigma^+ = \begin{bmatrix}
1/\sigma_1 & 0 & \cdots & 0 \\
0 & 1/\sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
$$

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])  # 3×2, no cuadrada

# Pseudoinversa
A_pinv = np.linalg.pinv(A)
print(f"Pseudoinversa: {A_pinv.shape}")  # 2×3

# Verificar propiedad: A @ A^+ @ A = A
print(np.allclose(A @ A_pinv @ A, A))  # True

# Resolver sistema sobredeterminado
b = np.array([1, 2, 3])
x = A_pinv @ b
print(f"Solución por mínimos cuadrados: {x}")
```

---

## 8. Comparación: SVD vs Eigendecomposición

```python
import numpy as np

# Matriz simétrica
A_sym = np.array([[4, 1],
                  [1, 3]])

print("=== MATRIZ SIMÉTRICA ===")

# Eigendecomposición
eigenvalues, eigenvectors = np.linalg.eig(A_sym)
print(f"Eigenvalores: {eigenvalues}")

# SVD
U, S, VT = np.linalg.svd(A_sym)
print(f"Valores singulares: {S}")
print(f"¿Son iguales? {np.allclose(np.abs(eigenvalues), S)}")

# Matriz no simétrica
A_nonsym = np.array([[3, 1],
                     [0, 2]])

print("\n=== MATRIZ NO SIMÉTRICA ===")

# Eigendecomposición
eigenvalues, eigenvectors = np.linalg.eig(A_nonsym)
print(f"Eigenvalores: {eigenvalues}")

# SVD
U, S, VT = np.linalg.svd(A_nonsym)
print(f"Valores singulares: {S}")
print(f"¿Son diferentes? {not np.allclose(np.abs(eigenvalues), S)}")
```

---

## 9. Eficiencia Computacional

### 9.1 Complejidad

**SVD completo:**
- Complejidad: $O(\min(m^2n, mn^2))$
- Para $m \times n$ con $m >> n$: $O(mn^2)$

**SVD truncado (randomizado):**
- Mucho más rápido para k pequeño
- Complejidad: $O(mnk)$

```python
from sklearn.decomposition import TruncatedSVD
import time

# Matriz grande
m, n = 10000, 1000
A = np.random.randn(m, n)

# SVD completo (lento)
start = time.time()
U, S, VT = np.linalg.svd(A, full_matrices=False)
tiempo_completo = time.time() - start

# SVD truncado (rápido)
k = 50
start = time.time()
svd = TruncatedSVD(n_components=k, random_state=42)
A_transformed = svd.fit_transform(A)
tiempo_truncado = time.time() - start

print(f"SVD completo: {tiempo_completo:.2f}s")
print(f"SVD truncado (k={k}): {tiempo_truncado:.2f}s")
print(f"Speedup: {tiempo_completo/tiempo_truncado:.1f}x")
```

### 9.2 Sparse SVD

Para matrices dispersas (sparse), usar bibliotecas especializadas:

```python
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import svds

# Matriz sparse (95% ceros)
A_sparse = sparse_random(5000, 1000, density=0.05)

# SVD para matrices sparse (k componentes)
U, S, VT = svds(A_sparse, k=10)

print(f"Valores singulares: {S}")
```

---

## 10. Errores Comunes

### ❌ Error 1: full_matrices en matrices grandes

```python
# ❌ Usa mucha memoria innecesariamente
U, S, VT = np.linalg.svd(A, full_matrices=True)  # U es m×m

# ✅ Más eficiente
U, S, VT = np.linalg.svd(A, full_matrices=False)  # U es m×min(m,n)
```

### ❌ Error 2: No centrar datos para PCA

```python
# ❌ Resultados incorrectos
U, S, VT = np.linalg.svd(X)

# ✅ Centrar primero
X_centered = X - np.mean(X, axis=0)
U, S, VT = np.linalg.svd(X_centered)
```

### ❌ Error 3: Confundir orden de valores singulares

```python
# Los valores singulares ya vienen ordenados (mayor a menor)
U, S, VT = np.linalg.svd(A)
assert np.all(S[:-1] >= S[1:])  # Verificar orden decreciente
```

---

## 11. Ejercicios Prácticos

### Ejercicio 1: Compresión de Imagen
Carga una imagen y comprimela usando SVD con diferentes rangos (k=10, 50, 100). Calcula el ratio de compresión y el error.

### Ejercicio 2: Sistema de Recomendación
Crea una matriz de ratings 10×8 (usuarios×películas) con valores faltantes. Usa SVD para predecir las calificaciones faltantes.

### Ejercicio 3: PCA Manual
Implementa PCA usando SVD desde cero. Compara con `sklearn.decomposition.PCA`.

### Ejercicio 4: Denoising
Genera una señal seno + ruido gaussiano. Usa SVD para eliminar el ruido.

---

## 📌 Resumen Clave

| Aspecto | Detalle |
|---------|---------|
| **Fórmula** | $A = U\Sigma V^T$ |
| **Aplicable a** | Cualquier matriz (m×n) |
| **Valores singulares** | Siempre reales ≥ 0, ordenados |
| **Interpretación** | Rotación + Escalado + Rotación |
| **Complejidad** | $O(\min(m^2n, mn^2))$ |
| **Aplicaciones ML** | PCA, recomendaciones, LSA, compresión |
| **Python** | `np.linalg.svd(A)` |

---

## 🎯 Próximos Pasos

**Día 6:** Análisis de Componentes Principales (PCA)
- Matemáticas detrás de PCA
- PCA usando SVD
- Aplicaciones en reducción de dimensionalidad
- Visualización de datos de alta dimensión

---

*SVD es una de las herramientas más poderosas en álgebra lineal y ML. ¡Dominarla te abre las puertas a técnicas avanzadas!*
