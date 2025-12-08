# 📐 Álgebra Lineal Avanzada - Semanas 3 y 4
## Eigenvalues, SVD y PCA

## 🎯 Objetivos

- Calcular determinantes e inversas
- Entender eigenvalues y eigenvectors
- Aplicar descomposición SVD
- Implementar PCA desde cero
- Aplicación real: Compresión de imágenes

---

## 📚 Contenido por Día

### **Día 1: Determinantes**

**Teoría**:
- ¿Qué es un determinante?
- Interpretación geométrica (área/volumen)
- Cálculo de determinantes 2x2, 3x3, nxn
- Propiedades

**Aplicación en IA**: Determinar si una matriz es invertible

### **Día 2: Matrices Inversas**

**Teoría**:
- Definición de matriz inversa
- Cálculo de inversas
- Pseudo-inversas
- Método de Gauss-Jordan

**Aplicación en IA**: Resolver sistemas lineales en regresión

### **Día 3-4: Eigenvalues y Eigenvectors**

**Teoría**:
- Definición: $Av = \lambda v$
- Cálculo de eigenvalues
- Cálculo de eigenvectors
- Diagonalización

**Aplicación en IA**: 
- PCA usa eigenvectors
- Análisis de estabilidad en sistemas
- Spectral clustering

### **Día 5: Descomposición SVD**

**Teoría**:
- Singular Value Decomposition: $A = U\Sigma V^T$
- Cálculo de SVD
- Interpretación geométrica
- Reducción de rango

**Aplicación en IA**:
- Sistemas de recomendación
- Compresión de imágenes
- Reducción de dimensionalidad

### **Día 6: PCA (Principal Component Analysis)**

**Teoría**:
- Reducción de dimensionalidad
- Encontrar componentes principales
- Varianza explicada
- Implementación desde cero

**Aplicación en IA**:
- Feature engineering
- Visualización de datos alta dimensión
- Preprocessing

### **Día 7: PROYECTO - Compresión de Imágenes**

**Objetivo**: Usar SVD para comprimir imágenes

**Tareas**:
1. Cargar imagen como matriz
2. Aplicar SVD
3. Reconstruir con k valores singulares
4. Comparar compresión vs calidad
5. Visualizar resultados

---

## 💻 Ejercicios Principales

### Ejercicio 1: Eigenvalues de una Matriz de Covarianza
```python
import numpy as np

# Matriz de covarianza de datos
cov_matrix = np.array([[4, 2], [2, 3]])

# TODO: Calcula eigenvalues y eigenvectors
# eigenvalues, eigenvectors = ...

# Interpreta: ¿Qué dirección tiene mayor varianza?
```

### Ejercicio 2: PCA Paso a Paso
```python
def pca_manual(X, n_components=2):
    """
    Implementa PCA desde cero.
    
    Pasos:
    1. Centrar datos (restar media)
    2. Calcular matriz de covarianza
    3. Encontrar eigenvalues/eigenvectors
    4. Ordenar por eigenvalues
    5. Proyectar datos
    """
    # TODO: Implementa esto
    pass
```

### Ejercicio 3: Compresión con SVD
```python
from PIL import Image

def compress_image(image_path, k):
    """
    Comprime imagen usando SVD.
    
    Args:
        image_path: Ruta a la imagen
        k: Número de valores singulares a mantener
    """
    # Cargar imagen
    img = np.array(Image.open(image_path).convert('L'))
    
    # TODO: Aplicar SVD
    # U, S, Vt = np.linalg.svd(img, full_matrices=False)
    
    # TODO: Reconstruir con k valores
    # img_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    return img_compressed
```

---

## 🎯 Proyecto Final: Análisis PCA de Dataset

**Dataset sugerido**: Iris, Wine, o MNIST simplificado

**Tareas**:
1. Cargar dataset multidimensional
2. Aplicar PCA para reducir a 2D
3. Visualizar en 2D con colores por clase
4. Analizar varianza explicada
5. Comparar clasificación en espacio original vs PCA

**Entregables**:
- Código Python funcionando
- Visualizaciones
- Análisis de cuánta información se retiene
- README explicativo

---

## 🔑 Conceptos Clave

### Eigenvalues/Eigenvectors

Un **eigenvector** de una matriz $A$ es un vector $v$ que solo cambia de escala al multiplicarlo por $A$:

$$Av = \lambda v$$

Donde $\lambda$ es el **eigenvalue** (factor de escala).

**Intuición**: Direcciones que la transformación solo estira/comprime.

### SVD (Descomposición en Valores Singulares)

Cualquier matriz $A_{m \times n}$ se puede descomponer en:

$$A = U\Sigma V^T$$

Donde:
- $U$: Eigenvectors de $AA^T$ (espaciorow)
- $\Sigma$: Valores singulares (raíces de eigenvalues)
- $V^T$: Eigenvectors de $A^TA$ (espacio columna)

### PCA

Encuentra las direcciones de **máxima varianza** en los datos.

**Algoritmo**:
1. Centrar datos: $X_{centered} = X - \text{mean}(X)$
2. Matriz de covarianza: $C = \frac{1}{n}X^TX$
3. Eigenvalues/eigenvectors de $C$
4. Proyectar: $X_{PCA} = X \cdot \text{eigenvectors}$

---

## ✅ Checklist de Progreso

### Conceptos
- [ ] Entiendo qué es un determinante
- [ ] Sé calcular inversas de matrices
- [ ] Comprendo eigenvalues/eigenvectors
- [ ] Entiendo SVD y sus componentes
- [ ] Sé qué es PCA y para qué sirve

### Implementación
- [ ] Calculé eigenvalues con NumPy
- [ ] Implementé PCA desde cero
- [ ] Usé SVD para compresión
- [ ] Visualicé resultados de PCA

### Proyecto
- [ ] Comprimí imágenes con SVD
- [ ] Analicé trade-off compresión/calidad
- [ ] Apliqué PCA a dataset real
- [ ] Documenté resultados

---

## 📚 Recursos

### Videos
- **3Blue1Brown**: "Eigenvalues and Eigenvectors"
- **StatQuest**: "PCA Clearly Explained"
- **Computerphile**: "Singular Value Decomposition"

### Lectura
- Capítulo 7: Eigenvalues - *Linear Algebra and Its Applications*
- Capítulo 10: SVD - *Introduction to Linear Algebra*

### Interactivos
- [Visualizando Eigenvectors](http://setosa.io/ev/eigenvectors-and-eigenvalues/)
- [PCA Explicado Visualmente](http://setosa.io/ev/principal-component-analysis/)

---

## 💡 Conexión con IA

### PCA en Feature Engineering
```python
from sklearn.decomposition import PCA

# Reducir 784 features (MNIST) a 50
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_train)

# Retiene ~95% de información con 13x menos features!
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")
```

### SVD en Recomendaciones
```python
# Matriz usuarios x películas
# SVD encuentra factores latentes (géneros implícitos)
U, S, Vt = np.linalg.svd(ratings_matrix)

# Reconstruir con k factores
k = 20
recommendations = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

---

**¡Esta es la base matemática de muchos algoritmos de ML!** 🚀

**Siguiente**: Cálculo y Optimización
