# 🧮 Machine Learning Básico - Regresión Lineal

## 🎯 Objetivos de Aprendizaje

Al finalizar esta semana, podrás:
- Entender qué es la regresión lineal y cuándo usarla
- Implementar regresión lineal desde cero
- Usar scikit-learn para regresión
- Evaluar modelos con métricas apropiadas
- Visualizar resultados y diagnósticos

## 📚 Contenido

### 1. Teoría Fundamental

#### ¿Qué es Regresión Lineal?

La **regresión lineal** es un algoritmo para predecir valores continuos. Encuentra la mejor línea que se ajusta a los datos.

**Ejemplo**: Predecir precio de casas basado en tamaño
- **Input (X)**: Metros cuadrados
- **Output (y)**: Precio
- **Modelo**: Encuentra una línea que relacione X con y

#### Ecuación

**Regresión lineal simple** (una variable):
$$y = mx + b$$

Donde:
- `y`: Variable a predecir (target)
- `x`: Variable independiente (feature)
- `m`: Pendiente (slope, weight)
- `b`: Intercepto (bias)

**Regresión lineal múltiple** (varias variables):
$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

O en forma vectorial:
$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$

#### ¿Cómo Encontrar m y b?

**Objetivo**: Minimizar el error entre predicciones y valores reales.

**Función de costo (MSE - Mean Squared Error)**:
$$J(w, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Métodos para optimizar**:
1. **Ecuación Normal** (solución cerrada)
2. **Descenso por Gradiente** (iterativo)

### 2. Implementación desde Cero

```python
import numpy as np

class LinearRegression:
    """Regresión lineal implementada desde cero"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        """
        Entrena el modelo usando descenso por gradiente.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
            learning_rate: Tasa de aprendizaje
            epochs: Número de iteraciones
        """
        n_samples, n_features = X.shape
        
        # Inicializar parámetros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Descenso por gradiente
        for epoch in range(epochs):
            # Predicción
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calcular gradientes
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Actualizar parámetros
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            # Opcional: imprimir progreso
            if epoch % 100 == 0:
                mse = np.mean((y_pred - y) ** 2)
                print(f"Epoch {epoch}, MSE: {mse:.4f}")
    
    def predict(self, X):
        """Hace predicciones"""
        return np.dot(X, self.weights) + self.bias
```

### 3. Usando scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Datos de ejemplo
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Evaluar
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Coeficientes: {model.coef_}")
print(f"Intercepto: {model.intercept_}")
```

### 4. Métricas de Evaluación

#### Mean Squared Error (MSE)
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- Penaliza más los errores grandes
- Siempre positivo
- Mismo unidad que y²

#### Root Mean Squared Error (RMSE)
$$RMSE = \sqrt{MSE}$$

- Misma unidad que y
- Más interpretable que MSE

#### Mean Absolute Error (MAE)
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- Menos sensible a outliers
- Más robusto que MSE

#### R² (Coeficiente de Determinación)
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- Rango: [0, 1] (puede ser negativo si el modelo es muy malo)
- R² = 1: Ajuste perfecto
- R² = 0: Modelo tan bueno como predecir la media
- Mide qué % de varianza explica el modelo

### 5. Visualización

```python
import matplotlib.pyplot as plt

def plot_regression(X, y, y_pred):
    """Visualiza regresión lineal"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot de datos reales
    plt.scatter(X, y, color='blue', label='Datos reales', alpha=0.6)
    
    # Línea de regresión
    plt.plot(X, y_pred, color='red', linewidth=2, label='Predicción')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Regresión Lineal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_residuals(y_true, y_pred):
    """Visualiza residuos (errores)"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Residuos vs predicciones
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title('Residuos vs Predicciones')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Histograma de residuos
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Residuos')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## 📝 Ejercicios Prácticos

### Ejercicio 1: Implementación Básica
Implementa regresión lineal desde cero para predecir temperaturas.

### Ejercicio 2: Múltiples Features
Usa regresión lineal múltiple para predecir precios de casas con varios features.

### Ejercicio 3: Comparación de Métricas
Compara MSE, RMSE, MAE y R² en diferentes datasets.

### Ejercicio 4: Diagnóstico
Analiza residuos para detectar problemas en el modelo.

## 🎯 Mini-Proyecto: Predictor de Salarios

**Objetivo**: Predecir salarios basado en años de experiencia.

**Dataset**: `salaries.csv` (proporcionado)

**Tareas**:
1. Cargar y explorar datos
2. Visualizar relación entre experiencia y salario
3. Entrenar modelo de regresión lineal
4. Evaluar con todas las métricas
5. Visualizar línea de regresión
6. Analizar residuos
7. Hacer predicciones para nuevos datos

## 💡 Tips y Buenas Prácticas

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Validación Cruzada
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"R² promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Detección de Overfitting
```python
# Compara error en train vs test
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"R² Train: {train_score:.4f}")
print(f"R² Test: {test_score:.4f}")

if train_score - test_score > 0.1:
    print("⚠️ Posible overfitting")
```

## 🔍 Cuando NO usar Regresión Lineal

- Relación no lineal entre variables
- Variables categóricas (usa regresión logística)
- Muchos outliers (usa modelos robustos)
- Multicolinealidad alta (features muy correlacionados)

## 📚 Recursos Adicionales

- StatQuest: "Linear Regression" (YouTube)
- Libro: "Introduction to Statistical Learning"
- Documentación scikit-learn

## ✅ Checklist de Progreso

- [ ] Entiendo la ecuación de regresión lineal
- [ ] Puedo implementar regresión desde cero
- [ ] Sé usar scikit-learn para regresión
- [ ] Conozco las métricas de evaluación
- [ ] Puedo interpretar R²
- [ ] Sé analizar residuos
- [ ] Completé el proyecto de salarios

---

**Siguiente tema**: Regresión Polinomial y Regularización
