# Día 7: Aplicaciones Avanzadas de Álgebra Lineal en Machine Learning

## 📋 Objetivos del Día
- Integrar todos los conceptos de álgebra lineal avanzada
- Implementar proyectos completos usando eigenvalores, SVD y PCA
- Reconocer patrones algebraicos en algoritmos de ML modernos
- Aplicar técnicas avanzadas en problemas reales
- Prepararse para Deep Learning con fundamentos sólidos

---

## 1. Reconocimiento Facial con Eigenfaces

### 1.1 Concepto

**Eigenfaces** es una de las primeras aplicaciones exitosas de PCA en Computer Vision (Turk & Pentland, 1991).

**Idea principal:**
- Cada imagen de rostro es un vector de alta dimensión (ej. 100×100 = 10,000 dimensiones)
- PCA encuentra las "caras principales" (eigenfaces)
- Cualquier rostro se puede representar como combinación de eigenfaces

### 1.2 Implementación Completa

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Cargar dataset de rostros
print("Cargando dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data  # (n_samples, h*w)
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(f"Total de ejemplos: {n_samples}")
print(f"Dimensión de imagen: {h}×{w} = {h*w} píxeles")
print(f"Clases (personas): {n_classes}")

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. PCA para extraer eigenfaces
print("\nExtrayendo eigenfaces...")
n_components = 150  # Número de eigenfaces

pca = PCA(n_components=n_components, whiten=True, random_state=42)
pca.fit(X_train)

# Eigenfaces = componentes principales
eigenfaces = pca.components_.reshape((n_components, h, w))

# 4. Proyectar datos al espacio de eigenfaces
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Dimensión original: {X_train.shape[1]}")
print(f"Dimensión reducida: {X_train_pca.shape[1]}")
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")

# 5. Entrenar clasificador en espacio PCA
print("\nEntrenando clasificador SVM...")
clf = SVC(kernel='rbf', class_weight='balanced', gamma='auto')
clf.fit(X_train_pca, y_train)

# 6. Predicciones
y_pred = clf.predict(X_test_pca)

# 7. Evaluación
print("\n" + "="*50)
print("RESULTADOS")
print("="*50)
print(classification_report(y_test, y_pred, target_names=target_names))

# 8. Visualización de eigenfaces
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Visualizar galería de imágenes"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# Mostrar primeras 12 eigenfaces
eigenface_titles = [f"Eigenface {i+1}" for i in range(12)]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.suptitle("Primeras 12 Eigenfaces", size=16)
plt.show()

# 9. Reconstrucción de rostros
def reconstruct_face(original, n_components_list):
    """Reconstruir rostro con diferentes números de componentes"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original
    axes[0].imshow(original.reshape(h, w), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Reconstrucciones con diferentes k
    for idx, k in enumerate(n_components_list, 1):
        pca_k = PCA(n_components=k)
        pca_k.fit(X_train)
        
        # Proyectar y reconstruir
        projected = pca_k.transform(original.reshape(1, -1))
        reconstructed = pca_k.inverse_transform(projected)
        
        # Calcular error
        error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
        variance = pca_k.explained_variance_ratio_.sum()
        
        axes[idx].imshow(reconstructed.reshape(h, w), cmap='gray')
        axes[idx].set_title(f'k={k}\nError: {error:.2%}\nVarianza: {variance:.1%}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Ejemplo de reconstrucción
test_face = X_test[0]
reconstruct_face(test_face, [10, 30, 50, 75, 150])

# 10. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusión - Reconocimiento Facial')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

### 1.3 Análisis de Resultados

```python
# Varianza explicada por componente
plt.figure(figsize=(12, 5))

# Varianza individual
plt.subplot(1, 2, 1)
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.xlabel('Componente')
plt.ylabel('Varianza explicada')
plt.title('Varianza por Componente')
plt.grid()

# Varianza acumulada
plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.title('Varianza Acumulada')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Cuántos componentes necesitamos?
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_comp = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold) + 1
    print(f"{threshold:.0%} varianza → {n_comp} componentes " 
          f"({n_comp/h/w:.1%} de dimensión original)")
```

---

## 2. Compresión de Imágenes con SVD

### 2.1 Pipeline Completo

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error

class ImageCompressorSVD:
    """Compresor de imágenes usando SVD"""
    
    def __init__(self, image_path):
        """Cargar imagen"""
        self.img_original = np.array(Image.open(image_path))
        
        # Verificar si es color o gris
        if len(self.img_original.shape) == 3:
            self.is_color = True
            self.height, self.width, self.channels = self.img_original.shape
        else:
            self.is_color = False
            self.height, self.width = self.img_original.shape
            self.channels = 1
    
    def compress(self, k):
        """
        Comprimir imagen manteniendo k valores singulares
        """
        if self.is_color:
            # Procesar cada canal RGB por separado
            compressed_channels = []
            for c in range(3):
                U, S, VT = np.linalg.svd(self.img_original[:, :, c], full_matrices=False)
                compressed = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
                compressed_channels.append(compressed)
            
            compressed_img = np.stack(compressed_channels, axis=2)
            compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)
        else:
            # Imagen en escala de grises
            U, S, VT = np.linalg.svd(self.img_original, full_matrices=False)
            compressed_img = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
            compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)
        
        return compressed_img
    
    def compression_ratio(self, k):
        """Calcular ratio de compresión"""
        original_size = self.height * self.width * self.channels
        
        if self.is_color:
            # Para cada canal: U[:, :k] + S[:k] + VT[:k, :]
            compressed_size = 3 * (self.height * k + k + k * self.width)
        else:
            compressed_size = self.height * k + k + k * self.width
        
        return original_size / compressed_size
    
    def calculate_metrics(self, compressed_img):
        """Calcular métricas de calidad"""
        mse = mean_squared_error(
            self.img_original.flatten(), 
            compressed_img.flatten()
        )
        
        # PSNR (Peak Signal-to-Noise Ratio)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return {'MSE': mse, 'PSNR': psnr}
    
    def analyze_compression(self, k_values):
        """Analizar diferentes niveles de compresión"""
        results = []
        
        for k in k_values:
            compressed = self.compress(k)
            ratio = self.compression_ratio(k)
            metrics = self.calculate_metrics(compressed)
            
            results.append({
                'k': k,
                'ratio': ratio,
                'mse': metrics['MSE'],
                'psnr': metrics['PSNR'],
                'image': compressed
            })
        
        return results
    
    def visualize_compression(self, k_values):
        """Visualizar resultados de compresión"""
        results = self.analyze_compression(k_values)
        
        n = len(k_values) + 1
        cols = 3
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.ravel() if n > 1 else [axes]
        
        # Imagen original
        axes[0].imshow(self.img_original, cmap='gray' if not self.is_color else None)
        axes[0].set_title(f'Original\n{self.height}×{self.width}')
        axes[0].axis('off')
        
        # Compresiones
        for idx, result in enumerate(results, 1):
            axes[idx].imshow(result['image'], cmap='gray' if not self.is_color else None)
            axes[idx].set_title(
                f"k={result['k']}\n"
                f"Compresión: {result['ratio']:.1f}×\n"
                f"PSNR: {result['psnr']:.1f} dB"
            )
            axes[idx].axis('off')
        
        # Ocultar ejes sobrantes
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Gráfica de métricas
        self._plot_metrics(results)
    
    def _plot_metrics(self, results):
        """Graficar métricas vs k"""
        k_vals = [r['k'] for r in results]
        ratios = [r['ratio'] for r in results]
        psnrs = [r['psnr'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Ratio de compresión
        ax1.plot(k_vals, ratios, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('k (componentes)', fontsize=12)
        ax1.set_ylabel('Ratio de compresión', fontsize=12)
        ax1.set_title('Compresión vs Componentes', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # PSNR (calidad)
        ax2.plot(k_vals, psnrs, 'o-', color='green', linewidth=2, markersize=8)
        ax2.axhline(y=30, color='r', linestyle='--', label='Calidad aceptable (30 dB)')
        ax2.set_xlabel('k (componentes)', fontsize=12)
        ax2.set_ylabel('PSNR (dB)', fontsize=12)
        ax2.set_title('Calidad vs Componentes', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Uso del compresor
compressor = ImageCompressorSVD('imagen.jpg')
compressor.visualize_compression(k_values=[5, 10, 20, 50, 100])
```

### 2.2 Análisis de Valores Singulares

```python
def analyze_singular_values(image_path):
    """Analizar distribución de valores singulares"""
    img = np.array(Image.open(image_path).convert('L'))
    
    # SVD
    U, S, VT = np.linalg.svd(img, full_matrices=False)
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Imagen original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    # 2. Valores singulares
    axes[0, 1].plot(S, 'o-')
    axes[0, 1].set_xlabel('Índice')
    axes[0, 1].set_ylabel('Valor singular')
    axes[0, 1].set_title('Valores Singulares')
    axes[0, 1].grid(True)
    
    # 3. Escala logarítmica
    axes[1, 0].semilogy(S, 'o-')
    axes[1, 0].set_xlabel('Índice')
    axes[1, 0].set_ylabel('Valor singular (log)')
    axes[1, 0].set_title('Valores Singulares (escala log)')
    axes[1, 0].grid(True)
    
    # 4. Energía acumulada
    energy = np.cumsum(S**2) / np.sum(S**2)
    axes[1, 1].plot(energy, 'o-')
    axes[1, 1].axhline(y=0.90, color='r', linestyle='--', label='90% energía')
    axes[1, 1].axhline(y=0.95, color='g', linestyle='--', label='95% energía')
    axes[1, 1].set_xlabel('Número de componentes')
    axes[1, 1].set_ylabel('Energía acumulada')
    axes[1, 1].set_title('Energía vs Componentes')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Reporte
    print("Análisis de Valores Singulares")
    print("="*50)
    print(f"Dimensiones imagen: {img.shape}")
    print(f"Rango: {np.linalg.matrix_rank(img)}")
    print(f"Valor singular máximo: {S[0]:.2f}")
    print(f"Valor singular mínimo: {S[-1]:.2e}")
    print(f"Condición: {S[0]/S[-1]:.2e}")
    
    for threshold in [0.90, 0.95, 0.99]:
        k = np.argmax(energy >= threshold) + 1
        print(f"\n{threshold:.0%} energía → {k} componentes ({k/len(S):.1%})")

analyze_singular_values('imagen.jpg')
```

---

## 3. Sistema de Recomendación con Matrix Factorization

### 3.1 Implementación con SVD

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class RecommenderSVD:
    """Sistema de recomendación usando SVD"""
    
    def __init__(self, n_factors=20, learning_rate=0.01, 
                 regularization=0.02, n_epochs=100):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, ratings_matrix, verbose=True):
        """
        Entrenar modelo usando SVD truncado
        ratings_matrix: (n_users, n_items)
        """
        # SVD truncado
        U, S, VT = np.linalg.svd(ratings_matrix, full_matrices=False)
        
        # Mantener solo k factores
        k = min(self.n_factors, len(S))
        self.user_factors = U[:, :k] @ np.diag(np.sqrt(S[:k]))
        self.item_factors = np.diag(np.sqrt(S[:k])) @ VT[:k, :]
        
        # Refinar con gradiente descendente
        self._refine_with_sgd(ratings_matrix, verbose)
        
        return self
    
    def _refine_with_sgd(self, R, verbose):
        """Refinar factorización con SGD"""
        n_users, n_items = R.shape
        
        # Máscara de valores observados
        mask = R > 0
        
        train_errors = []
        
        for epoch in range(self.n_epochs):
            # Para cada rating observado
            for i in range(n_users):
                for j in range(n_items):
                    if mask[i, j]:
                        # Predicción
                        pred = self.user_factors[i] @ self.item_factors[:, j]
                        error = R[i, j] - pred
                        
                        # Gradiente descendente
                        self.user_factors[i] += self.lr * (
                            error * self.item_factors[:, j] - 
                            self.reg * self.user_factors[i]
                        )
                        
                        self.item_factors[:, j] += self.lr * (
                            error * self.user_factors[i] - 
                            self.reg * self.item_factors[:, j]
                        )
            
            # Error en datos observados
            predictions = self.user_factors @ self.item_factors
            mse = mean_squared_error(R[mask], predictions[mask])
            train_errors.append(mse)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, MSE: {mse:.4f}")
        
        # Graficar convergencia
        plt.figure(figsize=(10, 5))
        plt.plot(train_errors)
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.title('Convergencia del Entrenamiento')
        plt.grid(True)
        plt.show()
    
    def predict(self, user_id, item_id):
        """Predecir rating para usuario-ítem específico"""
        return self.user_factors[user_id] @ self.item_factors[:, item_id]
    
    def predict_all(self):
        """Predecir todos los ratings"""
        return self.user_factors @ self.item_factors
    
    def recommend(self, user_id, n_recommendations=5, exclude_rated=True):
        """Recomendar ítems para un usuario"""
        # Predicciones para todos los ítems
        predictions = self.user_factors[user_id] @ self.item_factors
        
        # Ordenar por predicción descendente
        top_indices = np.argsort(predictions)[::-1]
        
        return top_indices[:n_recommendations], predictions[top_indices[:n_recommendations]]

# Ejemplo: MovieLens-like dataset
np.random.seed(42)

# Crear matriz de ratings (usuarios × películas)
n_users, n_movies = 100, 50

# Factores latentes verdaderos
true_user_factors = np.random.randn(n_users, 5)
true_movie_factors = np.random.randn(5, n_movies)

# Ratings verdaderos
true_ratings = true_user_factors @ true_movie_factors

# Normalizar a escala 1-5
true_ratings = 1 + 4 * (true_ratings - true_ratings.min()) / (true_ratings.max() - true_ratings.min())

# Simular matriz sparse (70% de ratings faltantes)
mask = np.random.rand(n_users, n_movies) > 0.7
observed_ratings = true_ratings.copy()
observed_ratings[~mask] = 0

print(f"Ratings observados: {mask.sum()} / {n_users * n_movies} ({mask.sum()/(n_users*n_movies):.1%})")

# Entrenar recomendador
recommender = RecommenderSVD(n_factors=10, n_epochs=50)
recommender.fit(observed_ratings)

# Evaluar
predictions = recommender.predict_all()
mse = mean_squared_error(true_ratings[mask], predictions[mask])
print(f"\nMSE en ratings observados: {mse:.4f}")

# Recomendar para usuario 0
user_id = 0
top_movies, scores = recommender.recommend(user_id, n_recommendations=5)

print(f"\nTop 5 recomendaciones para Usuario {user_id}:")
for idx, (movie, score) in enumerate(zip(top_movies, scores), 1):
    actual = true_ratings[user_id, movie]
    print(f"{idx}. Película {movie}: Predicho={score:.2f}, Real={actual:.2f}")
```

---

## 4. Reducción de Dimensionalidad para Visualización

### 4.1 Comparación: PCA vs t-SNE vs UMAP

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Cargar dataset
digits = load_digits()
X, y = digits.data, digits.target

print(f"Dimensión original: {X.shape}")
print(f"Clases: {np.unique(y)}")

# Aplicar diferentes técnicas
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X)

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scatter_params = {'s': 20, 'alpha': 0.6, 'cmap': 'tab10'}

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, **scatter_params)
axes[0].set_title(f'PCA\nVarianza explicada: {pca.explained_variance_ratio_.sum():.1%}')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, **scatter_params)
axes[1].set_title('t-SNE')
axes[1].set_xlabel('Dimensión 1')
axes[1].set_ylabel('Dimensión 2')

scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, **scatter_params)
axes[2].set_title('UMAP')
axes[2].set_xlabel('Dimensión 1')
axes[2].set_ylabel('Dimensión 2')

# Colorbar
plt.colorbar(scatter, ax=axes[2], label='Dígito')
plt.tight_layout()
plt.show()
```

### 4.2 Análisis de Clustering en Espacio Reducido

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

def analyze_clustering(X_2d, y_true, method_name):
    """Analizar calidad de clustering en espacio reducido"""
    # K-means en espacio 2D
    kmeans = KMeans(n_clusters=10, random_state=42)
    y_pred = kmeans.fit_predict(X_2d)
    
    # Métricas
    silhouette = silhouette_score(X_2d, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # Visualizar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Etiquetas verdaderas
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, 
                           s=20, alpha=0.6, cmap='tab10')
    ax1.set_title(f'{method_name} - Etiquetas Verdaderas')
    plt.colorbar(scatter1, ax=ax1)
    
    # Clusters predichos
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, 
                           s=20, alpha=0.6, cmap='tab10')
    ax2.scatter(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', 
                label='Centroides')
    ax2.set_title(f'{method_name} - Clusters K-means\n'
                  f'Silhouette: {silhouette:.3f}, ARI: {ari:.3f}')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return {'silhouette': silhouette, 'ari': ari}

# Analizar cada método
results = {}
for name, X_2d in [('PCA', X_pca), ('t-SNE', X_tsne), ('UMAP', X_umap)]:
    results[name] = analyze_clustering(X_2d, y, name)

# Comparación
print("\nComparación de métodos:")
print("="*50)
for method, metrics in results.items():
    print(f"{method:10s} | Silhouette: {metrics['silhouette']:.3f} | ARI: {metrics['ari']:.3f}")
```

---

## 5. Álgebra Lineal en Redes Neuronales

### 5.1 Forward Pass como Operaciones Matriciales

```python
import numpy as np

class NeuralNetworkAlgebra:
    """Red neuronal vista como operaciones de álgebra lineal"""
    
    def __init__(self, layers):
        """
        layers: lista con número de neuronas por capa
        Ej: [784, 128, 64, 10] para MNIST
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Inicializar pesos (Xavier initialization)
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, X, verbose=True):
        """
        Forward pass explicando álgebra lineal
        X: (batch_size, input_dim)
        """
        activations = [X]
        pre_activations = []
        
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            if verbose:
                print(f"\nCapa {layer_idx + 1}:")
                print(f"  Entrada: {activations[-1].shape}")
                print(f"  Pesos W: {W.shape}")
                print(f"  Bias b: {b.shape}")
            
            # Transformación lineal: Z = XW + b
            Z = activations[-1] @ W + b
            pre_activations.append(Z)
            
            if verbose:
                print(f"  Pre-activación Z = X @ W + b: {Z.shape}")
            
            # Función de activación (ReLU para capas ocultas, softmax para salida)
            if layer_idx < len(self.weights) - 1:
                A = np.maximum(0, Z)  # ReLU
                if verbose:
                    print(f"  Post-activación (ReLU): {A.shape}")
            else:
                # Softmax para última capa
                exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
                if verbose:
                    print(f"  Post-activación (Softmax): {A.shape}")
            
            activations.append(A)
        
        return activations, pre_activations
    
    def analyze_transformations(self, X):
        """Analizar transformaciones geométricas en cada capa"""
        activations, _ = self.forward(X, verbose=False)
        
        print("\nAnálisis de Transformaciones:")
        print("="*60)
        
        for i, A in enumerate(activations):
            if i == 0:
                print(f"Entrada (X): {A.shape}")
            else:
                print(f"\nCapa {i}:")
                print(f"  Forma: {A.shape}")
                print(f"  Rango: {np.linalg.matrix_rank(A)}")
                
                # Norma
                norm = np.linalg.norm(A, axis=1).mean()
                print(f"  Norma promedio: {norm:.4f}")
                
                # Dispersión
                std = A.std()
                print(f"  Desv. estándar: {std:.4f}")
                
                # Activaciones nulas (para ReLU)
                if i < len(activations) - 1:
                    sparsity = (A == 0).mean()
                    print(f"  Sparsity (zeros): {sparsity:.2%}")

# Ejemplo con datos sintéticos
np.random.seed(42)
X_batch = np.random.randn(32, 784)  # 32 imágenes de 28×28

# Red para MNIST
nn = NeuralNetworkAlgebra([784, 128, 64, 10])

# Forward pass detallado
print("="*60)
print("FORWARD PASS - ANÁLISIS ALGEBRAICO")
print("="*60)
activations, pre_activations = nn.forward(X_batch)

# Análisis geométrico
nn.analyze_transformations(X_batch)
```

### 5.2 Backpropagation como Cálculo de Gradientes

```python
def backprop_algebra_explained(nn, X, y_true):
    """
    Explicar backpropagation con álgebra lineal
    """
    batch_size = X.shape[0]
    
    # Forward pass
    activations, pre_activations = nn.forward(X, verbose=False)
    y_pred = activations[-1]
    
    print("\nBACKPROPAGATION - ANÁLISIS ALGEBRAICO")
    print("="*60)
    
    # Gradiente de la pérdida (Cross-entropy + Softmax)
    dZ = y_pred - y_true  # Gradiente simplificado
    print(f"\nGradiente en salida (dL/dZ): {dZ.shape}")
    
    # Backprop a través de cada capa
    gradients_W = []
    gradients_b = []
    
    for layer_idx in range(len(nn.weights) - 1, -1, -1):
        print(f"\n--- Capa {layer_idx + 1} (hacia atrás) ---")
        
        # Activación de la capa anterior
        A_prev = activations[layer_idx]
        print(f"Activación anterior: {A_prev.shape}")
        print(f"Gradiente actual (dZ): {dZ.shape}")
        
        # Gradientes de pesos y bias
        # dW = (1/m) * A_prev^T @ dZ
        dW = (1 / batch_size) * (A_prev.T @ dZ)
        db = (1 / batch_size) * np.sum(dZ, axis=0, keepdims=True)
        
        print(f"Gradiente de pesos (dW = A^T @ dZ / m): {dW.shape}")
        print(f"Gradiente de bias (db): {db.shape}")
        
        gradients_W.insert(0, dW)
        gradients_b.insert(0, db)
        
        # Propagar gradiente a capa anterior
        if layer_idx > 0:
            dA_prev = dZ @ nn.weights[layer_idx].T
            print(f"Gradiente a capa anterior (dA = dZ @ W^T): {dA_prev.shape}")
            
            # Aplicar derivada de ReLU
            dZ = dA_prev * (pre_activations[layer_idx - 1] > 0)
            print(f"Después de ReLU' (dZ): {dZ.shape}")
    
    return gradients_W, gradients_b

# One-hot encoding de etiquetas
y_true = np.zeros((32, 10))
y_true[np.arange(32), np.random.randint(0, 10, 32)] = 1

# Calcular gradientes
grads_W, grads_b = backprop_algebra_explained(nn, X_batch, y_true)
```

---

## 6. Proyecto Final: Análisis Completo de Dataset

### 6.1 Pipeline End-to-End

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

class DataAnalysisPipeline:
    """Pipeline completo de análisis con álgebra lineal"""
    
    def __init__(self, data, target=None):
        self.data = data
        self.target = target
        self.scaler = StandardScaler()
        self.pca = None
        self.clustering = None
        
    def preprocess(self):
        """Preprocesamiento"""
        print("1. PREPROCESAMIENTO")
        print("="*60)
        
        # Eliminar NaN
        self.data = self.data.dropna()
        
        # Normalizar
        self.data_scaled = self.scaler.fit_transform(self.data)
        
        print(f"Forma: {self.data_scaled.shape}")
        print(f"Media: {self.data_scaled.mean(axis=0)[:5]}...")  # Primeras 5
        print(f"Std: {self.data_scaled.std(axis=0)[:5]}...")
        
        return self
    
    def analyze_correlation(self):
        """Analizar matriz de correlación"""
        print("\n2. ANÁLISIS DE CORRELACIÓN")
        print("="*60)
        
        corr_matrix = np.corrcoef(self.data_scaled.T)
        
        # Eigenvalores de matriz de correlación
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        print(f"Eigenvalores: {eigenvalues[:10]}")
        print(f"Condición: {eigenvalues[0] / eigenvalues[-1]:.2e}")
        
        # Visualizar
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def apply_pca(self, n_components=None, variance_threshold=0.95):
        """Aplicar PCA"""
        print("\n3. PCA - REDUCCIÓN DE DIMENSIONALIDAD")
        print("="*60)
        
        if n_components is None:
            # Determinar n automáticamente
            pca_temp = PCA()
            pca_temp.fit(self.data_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            print(f"Componentes para {variance_threshold:.0%} varianza: {n_components}")
        
        # Aplicar PCA
        self.pca = PCA(n_components=n_components)
        self.data_pca = self.pca.fit_transform(self.data_scaled)
        
        print(f"Dimensión original: {self.data_scaled.shape[1]}")
        print(f"Dimensión reducida: {self.data_pca.shape[1]}")
        print(f"Varianza explicada: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        # Visualizar varianza
        self._plot_pca_variance()
        
        return self
    
    def _plot_pca_variance(self):
        """Visualizar varianza de PCA"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Varianza por componente
        ax1.bar(range(1, len(self.pca.explained_variance_ratio_) + 1),
                self.pca.explained_variance_ratio_)
        ax1.set_xlabel('Componente Principal')
        ax1.set_ylabel('Varianza Explicada')
        ax1.set_title('Varianza por Componente')
        
        # Varianza acumulada
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'o-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax2.set_xlabel('Número de Componentes')
        ax2.set_ylabel('Varianza Acumulada')
        ax2.set_title('Varianza Acumulada')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def clustering_analysis(self, n_clusters_range=range(2, 11)):
        """Análisis de clustering"""
        print("\n4. ANÁLISIS DE CLUSTERING")
        print("="*60)
        
        # Método del codo
        inertias = []
        silhouettes = []
        
        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data_pca)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.data_pca, labels))
        
        # Visualizar métricas
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(n_clusters_range, inertias, 'o-')
        ax1.set_xlabel('Número de Clusters')
        ax1.set_ylabel('Inercia')
        ax1.set_title('Método del Codo')
        ax1.grid(True)
        
        ax2.plot(n_clusters_range, silhouettes, 'o-')
        ax2.set_xlabel('Número de Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Mejor k por silhouette
        best_k = n_clusters_range[np.argmax(silhouettes)]
        print(f"Mejor k (por Silhouette): {best_k}")
        
        return best_k
    
    def visualize_2d(self, method='pca', clusters=None):
        """Visualizar en 2D"""
        print(f"\n5. VISUALIZACIÓN 2D ({method.upper()})")
        print("="*60)
        
        if method == 'pca':
            if self.data_pca.shape[1] >= 2:
                X_2d = self.data_pca[:, :2]
            else:
                pca_2d = PCA(n_components=2)
                X_2d = pca_2d.fit_transform(self.data_scaled)
        elif method == 'tsne':
            tsne = TSNE(n_components=2, random_state=42)
            X_2d = tsne.fit_transform(self.data_scaled)
        
        # Visualizar
        plt.figure(figsize=(10, 8))
        
        if clusters is not None:
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                                c=clusters, cmap='viridis', 
                                s=50, alpha=0.6)
            plt.colorbar(scatter, label='Cluster')
        elif self.target is not None:
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                                c=self.target, cmap='tab10', 
                                s=50, alpha=0.6)
            plt.colorbar(scatter, label='Clase')
        else:
            plt.scatter(X_2d[:, 0], X_2d[:, 1], s=50, alpha=0.6)
        
        plt.xlabel(f'{method.upper()} Dimensión 1')
        plt.ylabel(f'{method.upper()} Dimensión 2')
        plt.title(f'Visualización 2D - {method.upper()}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_full_analysis(self):
        """Ejecutar análisis completo"""
        self.preprocess()
        self.analyze_correlation()
        self.apply_pca()
        best_k = self.clustering_analysis()
        
        # Aplicar clustering final
        kmeans_final = KMeans(n_clusters=best_k, random_state=42)
        clusters = kmeans_final.fit_predict(self.data_pca)
        
        # Visualizaciones
        self.visualize_2d(method='pca', clusters=clusters)
        self.visualize_2d(method='tsne', clusters=clusters)
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO")
        print("="*60)
        
        return {
            'pca_model': self.pca,
            'clusters': clusters,
            'best_k': best_k
        }

# Ejemplo con dataset iris
from sklearn.datasets import load_iris
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

pipeline = DataAnalysisPipeline(X_iris, target=y_iris)
results = pipeline.run_full_analysis()
```

---

## 📌 Resumen del Módulo Completo

### Conceptos Clave Dominados

| Tema | Aplicación Principal |
|------|---------------------|
| **Determinantes** | Invertibilidad, volumen de transformaciones |
| **Matriz Inversa** | Solución de sistemas, criptografía |
| **Eigenvalores/vectores** | Estabilidad de sistemas, análisis espectral |
| **SVD** | Compresión, recomendaciones, denoising |
| **PCA** | Reducción dimensional, visualización |
| **Aplicaciones** | ML end-to-end, redes neuronales |

### Siguiente Nivel

**Preparación para Deep Learning:**
- ✅ Operaciones matriciales masivas (GPUs)
- ✅ Gradientes y backpropagation
- ✅ Inicialización de pesos
- ✅ Regularización (dropout, batch norm)
- ✅ Arquitecturas (atención, transformers)

---

*¡Felicitaciones! Has completado los fundamentos avanzados de álgebra lineal. Estos conceptos son la base de todo el Machine Learning moderno.*
