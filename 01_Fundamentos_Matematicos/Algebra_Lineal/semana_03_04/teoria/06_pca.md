# Día 6: Análisis de Componentes Principales (PCA)

## 📋 Objetivos del Día
- Comprender qué es PCA y por qué es fundamental en ML
- Derivar PCA matemáticamente usando covarianza y SVD
- Implementar PCA desde cero y con scikit-learn
- Visualizar datos de alta dimensión en 2D/3D
- Aplicar PCA en preprocesamiento, compresión y exploración de datos
- Reconocer limitaciones y alternativas de PCA

---

## 1. ¿Qué es PCA?

### 1.1 Concepto Intuitivo

**Principal Component Analysis (PCA)** es una técnica de reducción de dimensionalidad que:

1. **Encuentra las direcciones de máxima varianza** en los datos
2. **Proyecta los datos** en un espacio de menor dimensión
3. **Preserva la mayor cantidad de información posible**

**Analogía visual:**
Imagina fotografiar una nube de puntos 3D. PCA encuentra el mejor ángulo para que la foto 2D capture la mayor variabilidad de los datos.

### 1.2 Motivación

**Problemas que resuelve PCA:**
- 📉 **Maldición de dimensionalidad:** Reducir features mejora rendimiento
- 👁️ **Visualización:** Proyectar datos de alta dimensión a 2D/3D
- 🗜️ **Compresión:** Almacenar datos usando menos espacio
- 🧹 **Eliminación de ruido:** Componentes pequeñas = ruido
- 🔍 **Feature extraction:** Crear nuevas features más informativas

### 1.3 Aplicaciones Reales

| Dominio | Aplicación |
|---------|------------|
| **Computer Vision** | Eigenfaces para reconocimiento facial |
| **Genómica** | Análisis de expresión génica (miles de genes → 2-3 componentes) |
| **Finanzas** | Reducir correlaciones entre activos |
| **NLP** | Reducción de embeddings de palabras |
| **Biología** | Análisis de datos de secuenciación |
| **Ciencia de datos** | Preprocesamiento antes de clustering/clasificación |

---

## 2. Matemáticas de PCA

### 2.1 Formulación del Problema

**Objetivo:** Dado un dataset $X$ de $n$ muestras y $d$ features, encontrar $k$ direcciones (componentes principales) que maximicen la varianza proyectada.

$$
X = \begin{bmatrix}
— \mathbf{x}_1^T — \\
— \mathbf{x}_2^T — \\
\vdots \\
— \mathbf{x}_n^T —
\end{bmatrix}_{n \times d}
$$

**Componente principal:** Vector $\mathbf{w}$ unitario que maximiza:

$$
\max_{\|\mathbf{w}\|=1} \text{Var}(X\mathbf{w})
$$

### 2.2 Derivación Usando Matriz de Covarianza

**Paso 1:** Centrar los datos (media = 0)
$$
\bar{X} = X - \mathbf{1}\boldsymbol{\mu}^T
$$

Donde $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$

**Paso 2:** Calcular matriz de covarianza
$$
C = \frac{1}{n}\bar{X}^T\bar{X} \quad \text{(matriz } d \times d\text{)}
$$

**Paso 3:** Encontrar eigenvalores y eigenvectores de $C$
$$
C\mathbf{w}_i = \lambda_i\mathbf{w}_i
$$

**Paso 4:** Ordenar eigenvalores (mayor a menor)
$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d
$$

**Paso 5:** Las componentes principales son los eigenvectores
$$
\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k
$$

**Paso 6:** Proyectar datos
$$
Z = \bar{X}W_k
$$

Donde $W_k = [\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k]$ es la matriz $d \times k$.

### 2.3 Interpretación Geométrica

**Varianza explicada por la componente $i$:**
$$
\text{Var}(\mathbf{w}_i) = \lambda_i
$$

**Varianza total:**
$$
\text{Var}_{\text{total}} = \sum_{i=1}^d \lambda_i = \text{tr}(C)
$$

**Proporción de varianza explicada:**
$$
\frac{\lambda_i}{\sum_{j=1}^d \lambda_j}
$$

---

## 3. PCA Usando SVD

### 3.1 Relación PCA-SVD

PCA también se puede obtener directamente con SVD (más eficiente):

$$
\bar{X} = U\Sigma V^T
$$

**Componentes principales:** Columnas de $V$

**Ventajas de usar SVD:**
- ✅ No necesita calcular $\bar{X}^T\bar{X}$ (más estable numéricamente)
- ✅ Más eficiente para $n >> d$
- ✅ scikit-learn usa SVD internamente

### 3.2 Equivalencia Matemática

**Eigendecomposición de covarianza:**
$$
C = \frac{1}{n}\bar{X}^T\bar{X} = V\Lambda V^T
$$

**SVD de datos centrados:**
$$
\bar{X} = U\Sigma V^T
$$

$$
\Rightarrow C = \frac{1}{n}V\Sigma^T U^T U\Sigma V^T = V\left(\frac{\Sigma^2}{n}\right)V^T
$$

Por lo tanto:
$$
\lambda_i = \frac{\sigma_i^2}{n}
$$

---

## 4. Implementación desde Cero

### 4.1 PCA con Eigendecomposición

```python
import numpy as np
import matplotlib.pyplot as plt

class PCA_Eigen:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        
    def fit(self, X):
        """
        X: (n_samples, n_features)
        """
        # 1. Centrar datos
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. Matriz de covarianza
        cov_matrix = np.cov(X_centered.T)  # (n_features, n_features)
        
        # 3. Eigenvalores y eigenvectores
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. Ordenar por eigenvalor (mayor a menor)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Seleccionar k componentes
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        return self
    
    def transform(self, X):
        """Proyectar datos a espacio de componentes principales"""
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Reconstruir datos originales desde componentes"""
        return X_transformed @ self.components.T + self.mean
    
    def explained_variance_ratio(self):
        """Proporción de varianza explicada por cada componente"""
        return self.explained_variance / np.sum(self.explained_variance)

# Ejemplo de uso
np.random.seed(42)
# Datos 3D con correlación
mean = [0, 0, 0]
cov = [[3, 2, 1],
       [2, 2, 0.5],
       [1, 0.5, 1]]
X = np.random.multivariate_normal(mean, cov, size=200)

# Aplicar PCA
pca = PCA_Eigen(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Forma original: {X.shape}")
print(f"Forma reducida: {X_pca.shape}")
print(f"Componentes principales:\n{pca.components}")
print(f"Varianza explicada: {pca.explained_variance}")
print(f"Proporción varianza: {pca.explained_variance_ratio()}")
```

### 4.2 PCA con SVD (Más Eficiente)

```python
class PCA_SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.singular_values = None
        
    def fit(self, X):
        # 1. Centrar
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. SVD
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. Componentes principales (filas de VT)
        self.components = VT[:self.n_components].T
        self.singular_values = S[:self.n_components]
        
        # Varianza explicada
        self.explained_variance = (S ** 2) / (len(X) - 1)
        self.explained_variance = self.explained_variance[:self.n_components]
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Comparar con eigendecomposición
pca_svd = PCA_SVD(n_components=2)
X_pca_svd = pca_svd.fit_transform(X)

print("\n=== Comparación ===")
print(f"PCA Eigen: {X_pca[:3]}")
print(f"PCA SVD: {X_pca_svd[:3]}")
print(f"¿Iguales? {np.allclose(np.abs(X_pca), np.abs(X_pca_svd))}")
```

---

## 5. PCA con scikit-learn

### 5.1 Uso Básico

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd

# Cargar dataset Iris (4 features)
iris = load_iris()
X = iris.data
y = iris.target

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Información del modelo
print(f"Componentes principales:\n{pca.components_}")
print(f"Varianza explicada: {pca.explained_variance_}")
print(f"Proporción varianza: {pca.explained_variance_ratio_}")
print(f"Varianza acumulada: {np.cumsum(pca.explained_variance_ratio_)}")

# Visualizar
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                     edgecolors='k', s=100, alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
plt.title('Iris Dataset - PCA Projection')
plt.colorbar(scatter, label='Clase')
plt.grid(alpha=0.3)
plt.show()
```

### 5.2 Selección Automática de Componentes

```python
# Mantener 95% de la varianza
pca_auto = PCA(n_components=0.95)
X_pca_auto = pca_auto.fit_transform(X)

print(f"Componentes seleccionadas automáticamente: {pca_auto.n_components_}")
print(f"Varianza total capturada: {sum(pca_auto.explained_variance_ratio_):.2%}")

# Scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca_auto.explained_variance_ratio_) + 1),
        pca_auto.explained_variance_ratio_)
plt.xlabel('Componente Principal')
plt.ylabel('Proporción de Varianza Explicada')
plt.title('Scree Plot')
plt.xticks(range(1, len(pca_auto.explained_variance_ratio_) + 1))
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### 5.3 Análisis de Loadings

```python
# Crear DataFrame con loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=iris.feature_names
)

print("\nLoadings (contribución de cada feature):")
print(loadings)

# Visualizar loadings
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(loadings, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(loadings.columns)))
ax.set_yticks(range(len(loadings.index)))
ax.set_xticklabels(loadings.columns)
ax.set_yticklabels(loadings.index)
plt.colorbar(im)
plt.title('PCA Loadings Heatmap')
plt.tight_layout()
plt.show()
```

---

## 6. Aplicaciones Prácticas

### 6.1 Visualización de Datos de Alta Dimensión

```python
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Cargar MNIST (784 features)
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X_mnist = mnist.data[:1000].astype(float)  # Primeras 1000 imágenes
y_mnist = mnist.target[:1000].astype(int)

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mnist)

# PCA a 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(f"Varianza explicada en 2D: {sum(pca.explained_variance_ratio_):.2%}")

# Visualizar
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_mnist, cmap='tab10',
                     alpha=0.6, edgecolors='k', linewidth=0.5)
plt.colorbar(scatter, label='Dígito')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('MNIST Digits - PCA 2D Projection')
plt.grid(alpha=0.3)
plt.show()
```

### 6.2 Compresión de Imágenes

```python
from PIL import Image

# Cargar imagen
img = np.array(Image.open('imagen.jpg').convert('L'))  # Escala de grises
print(f"Tamaño original: {img.shape}")

# Aplicar PCA por filas (cada fila es una "muestra")
pca = PCA(n_components=50)
img_compressed = pca.fit_transform(img)
img_reconstructed = pca.inverse_transform(img_compressed)

# Calcular ratio de compresión
original_size = img.size
compressed_size = img_compressed.size + pca.components_.size + pca.mean_.size
compression_ratio = original_size / compressed_size

print(f"Tamaño comprimido: {img_compressed.shape}")
print(f"Ratio de compresión: {compression_ratio:.2f}x")
print(f"Varianza preservada: {sum(pca.explained_variance_ratio_):.2%}")

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(img_reconstructed, cmap='gray')
axes[1].set_title(f'Reconstruida ({pca.n_components} componentes)')
axes[1].axis('off')

axes[2].imshow(np.abs(img - img_reconstructed), cmap='hot')
axes[2].set_title('Error')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### 6.3 Eigenfaces (Reconocimiento Facial)

```python
from sklearn.datasets import fetch_lfw_people

# Cargar dataset de caras
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X_faces = faces.data
y_faces = faces.target
n_samples, n_features = X_faces.shape
h, w = faces.images.shape[1:]

print(f"Caras: {n_samples} imágenes de {h}×{w} ({n_features} píxeles)")

# PCA
n_components = 150
pca = PCA(n_components=n_components, whiten=True)
X_pca = pca.fit_transform(X_faces)

print(f"Reducido a {n_components} eigenfaces")
print(f"Varianza explicada: {sum(pca.explained_variance_ratio_):.2%}")

# Visualizar eigenfaces
eigenfaces = pca.components_.reshape((n_components, h, w))

fig, axes = plt.subplots(3, 8, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i < n_components:
        ax.imshow(eigenfaces[i], cmap='gray')
        ax.set_title(f'EF {i+1}')
    ax.axis('off')
plt.suptitle('Primeras 24 Eigenfaces')
plt.tight_layout()
plt.show()

# Reconstruir una cara
cara_original = X_faces[0].reshape(h, w)
cara_pca = pca.transform(X_faces[0:1])
cara_reconstruida = pca.inverse_transform(cara_pca).reshape(h, w)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(cara_original, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(cara_reconstruida, cmap='gray')
axes[1].set_title(f'Reconstruida ({n_components} componentes)')
axes[2].imshow(np.abs(cara_original - cara_reconstruida), cmap='hot')
axes[2].set_title('Error')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 6.4 Detección de Anomalías

```python
from sklearn.datasets import make_blobs

# Datos normales + outliers
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Reconstruir
X_reconstructed = pca.inverse_transform(X_pca)

# Error de reconstrucción
reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)

# Threshold para anomalías
threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold

# Visualizar
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=anomalies, cmap='coolwarm', 
           edgecolors='k', s=50, alpha=0.7)
plt.title('Datos Originales\n(Rojo = Anomalías detectadas)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(reconstruction_error, bins=50, edgecolor='k', alpha=0.7)
plt.axvline(threshold, color='r', linestyle='--', linewidth=2, 
           label=f'Threshold (95%): {threshold:.2f}')
plt.xlabel('Error de Reconstrucción')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Anomalías detectadas: {np.sum(anomalies)}/{len(X)}")
```

### 6.5 Preprocesamiento para Machine Learning

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Dataset de alta dimensión
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

print(f"Dimensión original: {X.shape}")

# Modelo sin PCA
rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
scores_original = cross_val_score(rf_original, X, y, cv=5)

# Modelo con PCA
pca = PCA(n_components=0.95)  # 95% varianza
X_pca = pca.fit_transform(X)
print(f"Dimensión reducida: {X_pca.shape} ({pca.n_components_} componentes)")

rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
scores_pca = cross_val_score(rf_pca, X_pca, y, cv=5)

# Comparar
print(f"\nAccuracy sin PCA: {scores_original.mean():.3f} ± {scores_original.std():.3f}")
print(f"Accuracy con PCA: {scores_pca.mean():.3f} ± {scores_pca.std():.3f}")
print(f"Reducción de features: {(1 - X_pca.shape[1]/X.shape[1]):.1%}")
```

---

## 7. Limitaciones y Alternativas

### 7.1 Limitaciones de PCA

❌ **Asume linealidad**
- Solo encuentra proyecciones lineales
- No captura relaciones no lineales

❌ **Sensible a escala**
- Features con mayor varianza dominan
- **Solución:** Estandarizar datos antes de PCA

❌ **No garantiza separabilidad de clases**
- Maximiza varianza, no discriminación entre clases
- **Alternativa:** Linear Discriminant Analysis (LDA)

❌ **Interpretabilidad**
- Componentes principales = combinaciones de features
- Difícil interpretar el significado

### 7.2 Cuándo NO Usar PCA

```python
# ❌ Datos categóricos
# PCA asume datos continuos

# ❌ Datos ya decorrelacionados
# PCA no aporta valor si features son independientes

# ❌ Cuando interpretabilidad es crítica
# Las componentes principales son difíciles de explicar
```

### 7.3 Alternativas

**Para datos no lineales:**
- **Kernel PCA:** PCA en espacio de features transformado
- **t-SNE:** Visualización de alta dimensión (no lineal)
- **UMAP:** Similar a t-SNE pero más rápido
- **Autoencoders:** Redes neuronales para reducción no lineal

**Para clasificación supervisada:**
- **LDA (Linear Discriminant Analysis):** Maximiza separación entre clases

**Para datos sparse:**
- **Sparse PCA:** Componentes con pocos features activos
- **Truncated SVD:** No requiere centrar datos (mejor para sparse)

```python
# Ejemplo: Kernel PCA
from sklearn.decomposition import KernelPCA

# Datos con estructura no lineal
from sklearn.datasets import make_circles
X_circles, y_circles = make_circles(n_samples=400, factor=0.3, noise=0.05)

# PCA lineal (no funciona bien)
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_circles)

# Kernel PCA (funciona mejor)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X_circles)

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
axes[0].set_title('Datos Originales')

axes[1].scatter(X_pca_linear[:, 0], X_pca_linear[:, 1], c=y_circles, cmap='viridis')
axes[1].set_title('PCA Lineal (no separa)')

axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_circles, cmap='viridis')
axes[2].set_title('Kernel PCA (separa mejor)')

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 8. Buenas Prácticas

### 8.1 Estandarizar Siempre

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ✅ Pipeline recomendado
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

X_transformed = pipeline.fit_transform(X)
```

### 8.2 Determinar Número de Componentes

**Método 1: Varianza acumulada**
```python
pca = PCA()
pca.fit(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.95) + 1

print(f"Componentes para 95% varianza: {n_components}")

# Visualizar
plt.plot(cumsum)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.grid()
plt.show()
```

**Método 2: Elbow method (Scree plot)**
```python
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, 'o-')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Scree Plot - Buscar "codo"')
plt.grid()
plt.show()
```

**Método 3: Kaiser criterion (eigenvalor > 1)**
```python
# Para datos estandarizados
eigenvalues = pca.explained_variance_
n_components_kaiser = np.sum(eigenvalues > 1)
print(f"Kaiser criterion: {n_components_kaiser} componentes")
```

### 8.3 Validación Cruzada con PCA

```python
from sklearn.model_selection import GridSearchCV

# Buscar mejor número de componentes
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', RandomForestClassifier())
])

param_grid = {
    'pca__n_components': [5, 10, 20, 30, 50]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X, y)

print(f"Mejor n_components: {grid.best_params_['pca__n_components']}")
print(f"Mejor accuracy: {grid.best_score_:.3f}")
```

---

## 9. Errores Comunes

### ❌ Error 1: No Estandarizar

```python
# ❌ Features con diferentes escalas
X = np.array([[1, 1000], [2, 2000], [3, 3000]])
pca = PCA(n_components=1)
pca.fit(X)
# PC1 dominada por feature 2 (mayor varianza absoluta)

# ✅ Estandarizar primero
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca.fit(X_scaled)
```

### ❌ Error 2: Aplicar PCA en Train y Test Separadamente

```python
# ❌ Incorrecto
pca_train = PCA(n_components=10)
X_train_pca = pca_train.fit_transform(X_train)

pca_test = PCA(n_components=10)
X_test_pca = pca_test.fit_transform(X_test)  # ¡DATA LEAKAGE!

# ✅ Correcto
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)  # Usar mismo PCA
```

### ❌ Error 3: Interpretar Componentes como Features Originales

```python
# ❌ "PC1 representa la altura"
# Las componentes son COMBINACIONES de todas las features

# ✅ Mirar loadings para entender contribuciones
print(pca.components_[0])  # Contribución de cada feature a PC1
```

---

## 10. Ejercicios Prácticos

### Ejercicio 1: Iris Dataset
Aplica PCA al dataset Iris. Visualiza en 2D y analiza qué features contribuyen más a cada PC.

### Ejercicio 2: Wine Dataset
Compara rendimiento de clasificación con y sin PCA en el wine dataset de sklearn.

### Ejercicio 3: Compresión de Imágenes
Carga una imagen y comprímela con diferentes números de componentes. Grafica error vs ratio de compresión.

### Ejercicio 4: PCA Incremental
Para datasets grandes que no caben en memoria, usa `IncrementalPCA` de sklearn.

### Ejercicio 5: Comparación con t-SNE
Compara visualizaciones 2D de MNIST usando PCA y t-SNE. ¿Cuál separa mejor los dígitos?

---

## 📌 Resumen Clave

| Aspecto | Detalle |
|---------|---------|
| **Objetivo** | Reducir dimensionalidad preservando varianza |
| **Método** | Eigendecomposición de covarianza o SVD |
| **Componentes** | Eigenvectores de matriz de covarianza |
| **Varianza** | Eigenvalores indican importancia |
| **Linealidad** | Solo proyecciones lineales |
| **Preprocesamiento** | Estandarizar datos SIEMPRE |
| **Aplicaciones** | Visualización, compresión, denoising, preprocesamiento |
| **Python** | `sklearn.decomposition.PCA` |

---

## 🎯 Próximos Pasos

**Día 7:** Aplicaciones Avanzadas de Álgebra Lineal
- Diagonalización de matrices
- Descomposición QR
- Métodos iterativos para eigenvalores
- Aplicaciones en optimización

---

*PCA es una de las técnicas más usadas en ciencia de datos. ¡Dominarla es esencial para análisis exploratorio y preprocesamiento!*
