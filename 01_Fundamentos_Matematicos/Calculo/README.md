# 📊 Cálculo y Optimización - Semanas 5 y 6

## 🎯 Objetivos

- Entender derivadas y su interpretación
- Calcular derivadas parciales
- Dominar la regla de la cadena
- Implementar descenso por gradiente
- Optimizar funciones desde cero

---

## 📅 Plan de las 2 Semanas

### **Semana 5: Derivadas y Gradientes**

#### Día 1: Derivadas Básicas
- Definición de derivada
- Interpretación geométrica (pendiente)
- Reglas de derivación
- Derivadas comunes

#### Día 2: Regla de la Cadena
- Composición de funciones
- Chain rule
- Aplicación: backpropagation

#### Día 3: Derivadas Parciales
- Funciones multivariables
- Derivadas parciales
- Notación $\frac{\partial f}{\partial x}$

#### Día 4: Gradientes
- Vector gradiente
- Dirección de máximo crecimiento
- Visualización de gradientes

### **Semana 6: Optimización**

#### Día 5: Descenso por Gradiente
- Idea fundamental
- Algoritmo de gradient descent
- Learning rate
- Convergencia

#### Día 6-7: PROYECTO - Optimizador
- Implementar gradient descent desde cero
- Aplicar a regresión lineal
- Visualizar el proceso
- Comparar learning rates

---

## 📚 Teoría Fundamental

### Derivada

La **derivada** mide la tasa de cambio instantánea:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Interpretación geométrica**: Pendiente de la recta tangente

### Reglas de Derivación

| Función | Derivada |
|---------|----------|
| $c$ (constante) | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |

**Regla de la suma**: $(f + g)' = f' + g'$
**Regla del producto**: $(fg)' = f'g + fg'$
**Regla de la cadena**: $(f(g(x)))' = f'(g(x)) \cdot g'(x)$

### Derivadas Parciales

Para funciones de varias variables $f(x, y)$:

$$\frac{\partial f}{\partial x} = \text{derivada respecto a } x \text{ (manteniendo } y \text{ constante)}$$

### Gradiente

El **gradiente** es un vector de todas las derivadas parciales:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]$$

**Propiedad clave**: El gradiente apunta en la dirección de máximo crecimiento.

### Descenso por Gradiente

Algoritmo para minimizar una función:

$$x_{new} = x_{old} - \alpha \nabla f(x_{old})$$

Donde $\alpha$ es el **learning rate** (tasa de aprendizaje).

---

## 💻 Implementaciones

### Derivada Numérica

```python
def derivative(f, x, h=1e-5):
    """Aproxima la derivada de f en x"""
    return (f(x + h) - f(x)) / h

# Ejemplo
f = lambda x: x**2
print(f"f'(3) ≈ {derivative(f, 3)}")  # Debería ser ~6
```

### Gradiente Numérico

```python
import numpy as np

def gradient(f, x, h=1e-5):
    """
    Calcula gradiente numérico de f en punto x.
    
    Args:
        f: función escalar que recibe vector
        x: punto donde evaluar (numpy array)
        h: paso para aproximación
        
    Returns:
        Vector gradiente
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        
        x_minus = x.copy()
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad
```

### Descenso por Gradiente

```python
def gradient_descent(f, grad_f, x_init, learning_rate=0.1, iterations=100):
    """
    Minimiza función f usando gradient descent.
    
    Args:
        f: función a minimizar
        grad_f: función que calcula el gradiente
        x_init: punto inicial
        learning_rate: tasa de aprendizaje
        iterations: número de iteraciones
        
    Returns:
        x_min: punto mínimo encontrado
        history: historia de valores
    """
    x = x_init.copy()
    history = [x.copy()]
    
    for i in range(iterations):
        # Calcular gradiente
        grad = grad_f(x)
        
        # Actualizar parámetros
        x = x - learning_rate * grad
        
        history.append(x.copy())
        
        # Opcional: imprimir progreso
        if i % 10 == 0:
            print(f"Iter {i}: f(x) = {f(x):.4f}")
    
    return x, np.array(history)
```

---

## 🎯 Ejercicios Prácticos

### Ejercicio 1: Derivadas a Mano

Calcula las derivadas de:
1. $f(x) = 3x^2 + 2x + 1$
2. $g(x) = e^{2x}$
3. $h(x) = \ln(x^2 + 1)$
4. $k(x) = \sin(x^2)$ (usa regla de la cadena)

### Ejercicio 2: Implementa Derivadas Simbólicas

```python
class Expr:
    """Clase base para expresiones"""
    pass

class Var(Expr):
    """Variable x"""
    def derivative(self):
        return Const(1)

class Const(Expr):
    """Constante"""
    def __init__(self, value):
        self.value = value
    
    def derivative(self):
        return Const(0)

# TODO: Implementa Sum, Product, Power
# TODO: Implementa método derivative() para cada uno
```

### Ejercicio 3: Visualiza Descenso por Gradiente

```python
import matplotlib.pyplot as plt

def visualize_gradient_descent(f, grad_f, x_init, lr=0.1):
    """
    Visualiza el proceso de gradient descent.
    """
    # Ejecutar gradient descent
    x_min, history = gradient_descent(f, grad_f, x_init, lr, 50)
    
    # Crear gráfico
    x = np.linspace(-5, 5, 100)
    y = [f(np.array([xi])) for xi in x]
    
    plt.plot(x, y, 'b-', label='f(x)')
    plt.plot(history[:, 0], [f(h) for h in history], 
             'ro-', label='GD path', markersize=4)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gradient Descent (lr={lr})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Prueba con f(x) = x^2
f = lambda x: x[0]**2
grad_f = lambda x: np.array([2*x[0]])
visualize_gradient_descent(f, grad_f, np.array([4.0]), lr=0.1)
```

---

## 🏗️ PROYECTO: Regresión Lineal con Gradient Descent

### Objetivo
Implementar regresión lineal usando descenso por gradiente desde cero.

### Especificaciones

```python
class LinearRegressionGD:
    """Regresión lineal con gradient descent"""
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.history = []
    
    def fit(self, X, y):
        """
        Entrena el modelo.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Inicializar parámetros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.iterations):
            # Predicción
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calcular loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.history.append(loss)
            
            # Calcular gradientes
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Actualizar parámetros
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        """Hace predicciones"""
        return np.dot(X, self.weights) + self.bias
    
    def plot_loss(self):
        """Visualiza la convergencia"""
        plt.plot(self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        plt.show()
```

### Tareas del Proyecto

1. **Implementa la clase completa**
2. **Genera datos sintéticos**:
   ```python
   X = 2 * np.random.rand(100, 1)
   y = 4 + 3 * X + np.random.randn(100, 1)
   ```
3. **Entrena el modelo** con diferentes learning rates
4. **Visualiza**:
   - Datos y línea de regresión
   - Convergencia del loss
   - Comparación de learning rates
5. **Experimenta**:
   - ¿Qué pasa con lr muy grande?
   - ¿Qué pasa con lr muy pequeño?
   - ¿Cuántas iteraciones necesitas?

---

## ✅ Checklist

### Teoría
- [ ] Entiendo qué es una derivada
- [ ] Puedo calcular derivadas a mano
- [ ] Entiendo la regla de la cadena
- [ ] Sé qué son derivadas parciales
- [ ] Entiendo el gradiente y su interpretación

### Implementación
- [ ] Calculé derivadas numéricas
- [ ] Calculé gradientes numéricos
- [ ] Implementé gradient descent desde cero
- [ ] Visualicé el proceso de optimización

### Proyecto
- [ ] Regresión lineal con GD funciona
- [ ] Probé diferentes learning rates
- [ ] Visualicé convergencia
- [ ] Entiendo cuándo converge/diverge

---

## 🔗 Conexión con Deep Learning

### Backpropagation = Regla de la Cadena

```python
# Red neuronal simple: y = σ(Wx + b)
# Para entrenar, necesitamos ∂L/∂W

# Regla de la cadena:
# ∂L/∂W = ∂L/∂y · ∂y/∂z · ∂z/∂W
# donde z = Wx + b, y = σ(z), L = loss

# Esto es exactamente backpropagation!
```

### Optimizadores Modernos

El gradient descent básico evoluciona a:
- **SGD con Momentum**: Añade inercia
- **Adam**: Learning rate adaptativo
- **RMSprop**: Normaliza gradientes

Pero todos se basan en: **$x_{new} = x - \alpha \nabla f(x)$**

---

## 📚 Recursos

### Videos
- **3Blue1Brown**: "Essence of Calculus" (serie completa)
- **Khan Academy**: Differential Calculus

### Interactivos
- [Desmos Graphing Calculator](https://www.desmos.com/calculator)
- [Seeing Theory - Optimization](https://seeing-theory.brown.edu/)

---

**¡El cálculo es el corazón del Deep Learning!** 🧮

**Siguiente**: Probabilidad y Estadística
