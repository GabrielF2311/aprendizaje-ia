# 🐍 Python Avanzado para IA - Semana 14

## 🎯 Objetivos de la Semana

- Escribir código Pythonic y eficiente
- Dominar comprehensions y generators
- Entender decoradores
- Aplicar OOP en Machine Learning
- Mejores prácticas para proyectos de IA

## 💎 Por qué Python Avanzado

En ML/IA necesitas código que sea:
- **Eficiente**: Procesar grandes volúmenes de datos
- **Legible**: Colaborar con equipos
- **Mantenible**: Proyectos duran años
- **Profesional**: Estándares de la industria

---

## 📅 Plan de la Semana

### **Día 1-2: Comprehensions y Generators**

Código más limpio y eficiente:
- List comprehensions
- Dict comprehensions
- Set comprehensions
- Generator expressions
- Iteradores personalizados

💻 **Código**: `comprehensions.py`

---

### **Día 3: Decoradores**

Funcionalidad modular:
- Qué son los decoradores
- Crear decoradores simples
- Decoradores con parámetros
- Decoradores para ML (timing, logging, caching)

💻 **Código**: `decoradores.py`

---

### **Día 4-5: OOP para Machine Learning**

Programación orientada a objetos aplicada:
- Clases para modelos de ML
- Herencia y composición
- Métodos especiales (__init__, __call__, __repr__)
- Scikit-learn API design

💻 **Código**: `oop_para_ml.py`

---

### **Día 6: Mejores Prácticas**

Código profesional:
- PEP 8 style guide
- Type hints
- Docstrings
- Testing con pytest
- Logging

📖 **Guía**: `best_practices.md`

---

### **Día 7: PROYECTO - ML Utils Library**

Crea tu propia librería de utilidades:
- Data loaders
- Preprocesadores
- Evaluadores
- Visualizadores
- Todo con OOP y mejores prácticas

💻 **Código**: `proyecto_ml_utils/`

---

## 🔑 Conceptos Clave

### List Comprehensions

```python
# ❌ Forma básica
squares = []
for x in range(10):
    squares.append(x**2)

# ✅ Comprehension
squares = [x**2 for x in range(10)]

# ✅ Con condición
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Generators

```python
# ❌ Lista completa en memoria
def get_squares(n):
    return [x**2 for x in range(n)]

# ✅ Generator - lazy evaluation
def get_squares(n):
    for x in range(n):
        yield x**2

# Usa menos memoria para n grande
squares = get_squares(1_000_000)
```

### Decoradores Básicos

```python
import time

def timer(func):
    """Decorador para medir tiempo de ejecución"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f}s")
        return result
    return wrapper

@timer
def train_model(X, y):
    # Entrenamiento aquí
    pass
```

### OOP para ML

```python
class LinearRegression:
    """Modelo de regresión lineal siguiendo API de scikit-learn"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Entrena el modelo"""
        # Implementación
        return self
    
    def predict(self, X):
        """Hace predicciones"""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calcula R² score"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
```

---

## ✅ Checklist de Progreso

### Comprehensions
- [ ] Escribo list comprehensions naturalmente
- [ ] Uso dict comprehensions
- [ ] Entiendo cuándo usar generators
- [ ] Creo generators personalizados

### Decoradores
- [ ] Entiendo cómo funcionan
- [ ] Creo decoradores simples
- [ ] Uso decoradores con parámetros
- [ ] Aplico decoradores en ML (timing, logging)

### OOP
- [ ] Diseño clases siguiendo scikit-learn API
- [ ] Uso herencia apropiadamente
- [ ] Implemento métodos especiales
- [ ] Escribo código modular y reutilizable

### Mejores Prácticas
- [ ] Sigo PEP 8
- [ ] Uso type hints
- [ ] Escribo docstrings claros
- [ ] Escribo tests básicos
- [ ] Uso logging en lugar de prints

### Proyecto
- [ ] Librería funcional con múltiples módulos
- [ ] Código documentado
- [ ] Tests incluidos
- [ ] README profesional

---

## 💡 Tips de Python Avanzado

### 1. Comprehensions

```python
# ✅ Bueno: Legible
[x**2 for x in range(10) if x % 2 == 0]

# ❌ Malo: Demasiado complejo
[x**2 if x % 2 == 0 else x**3 for x in range(10) 
 if x > 5 or x < 2]
# Mejor usar un loop normal si es muy complejo
```

### 2. Generators para Datos Grandes

```python
def load_data_batches(filename, batch_size=1000):
    """Carga datos en batches - eficiente en memoria"""
    batch = []
    with open(filename) as f:
        for line in f:
            batch.append(process(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

### 3. Type Hints

```python
from typing import List, Tuple, Optional
import numpy as np

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.01
) -> Tuple[np.ndarray, float]:
    """
    Entrena un modelo.
    
    Args:
        X: Features (n_samples, n_features)
        y: Target (n_samples,)
        epochs: Número de epochs
        learning_rate: Tasa de aprendizaje
        
    Returns:
        Tuple de (weights, final_loss)
    """
    # Implementación
    pass
```

### 4. Context Managers

```python
# Para manejo de recursos
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"Elapsed: {self.end - self.start:.4f}s")

# Uso
with Timer():
    train_model(X, y)
```

---

## 📚 Recursos

### Documentación
- [PEP 8 - Style Guide](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)

### Libros
- **"Fluent Python"** - Luciano Ramalho
- **"Effective Python"** - Brett Slatkin
- **"Python Tricks"** - Dan Bader

### Videos
- **Corey Schafer** - OOP, Decorators (YouTube)
- **Raymond Hettinger** - Beyond PEP 8

---

## 🎯 Ejercicios Prácticos

### Comprehensions
Convierte estos loops a comprehensions:
```python
# 1. Números pares al cuadrado
result = []
for x in range(20):
    if x % 2 == 0:
        result.append(x**2)

# 2. Diccionario de letras a números
d = {}
for i, char in enumerate('abcdefg'):
    d[char] = i
```

### Decorador
Crea un decorador `@cache` que guarde resultados de funciones.

### Clase ML
Implementa una clase `KNNClassifier` siguiendo la API de scikit-learn.

---

## 🚀 Siguiente Paso

Empieza con `comprehensions.py` y practica código Pythonic!

**¡Escribe código profesional esta semana!** 🐍
