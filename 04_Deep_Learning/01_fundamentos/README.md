# 🧠 Deep Learning - Fundamentos de Redes Neuronales

## 🎯 Objetivos

- Entender la arquitectura de redes neuronales
- Implementar una red neuronal desde cero
- Comprender forward propagation y backpropagation
- Usar PyTorch para crear modelos

## 📚 Conceptos Fundamentales

### El Perceptrón

El **perceptrón** es la unidad básica de una red neuronal.

```
Inputs → [Weights] → Activation → Output
x₁ ────┐
x₂ ────┤─→ Σ(wᵢxᵢ + b) ─→ f(z) ─→ ŷ
x₃ ────┘
```

**Ecuación**:
$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$
$$\hat{y} = f(z)$$

Donde `f` es una **función de activación**.

### Funciones de Activación

#### Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Rango: (0, 1)
- Útil para probabilidades
- Problema: vanishing gradient

#### ReLU (Rectified Linear Unit)
$$\text{ReLU}(x) = \max(0, x)$$

- Rango: [0, ∞)
- Más usada en capas ocultas
- Rápida de computar

#### Tanh
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- Rango: (-1, 1)
- Centrada en cero

#### Softmax (para clasificación multi-clase)
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

- Output: probabilidades que suman 1

### Red Neuronal Multicapa

```
Input Layer → Hidden Layer(s) → Output Layer

x₁ ───┐
x₂ ───┤─→ [h₁, h₂, h₃] ─→ [o₁, o₂]
x₃ ───┘
```

**Notación**:
- **L**: Número total de capas
- **n[l]**: Número de neuronas en capa l
- **W[l]**: Matriz de pesos de capa l
- **b[l]**: Vector de bias de capa l

### Forward Propagation

Proceso de calcular la salida de la red:

```python
# Capa 1
Z[1] = W[1] @ X + b[1]
A[1] = relu(Z[1])

# Capa 2
Z[2] = W[2] @ A[1] + b[2]
A[2] = sigmoid(Z[2])

# Output
ŷ = A[2]
```

### Backward Propagation

Proceso de calcular gradientes para actualizar pesos:

**Regla de la cadena**:
$$\frac{\partial J}{\partial W^{[l]}} = \frac{\partial J}{\partial A^{[l]}} \cdot \frac{\partial A^{[l]}}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}}$$

## 💻 Implementación desde Cero

```python
import numpy as np

class NeuralNetwork:
    """Red neuronal de 2 capas desde cero"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar pesos aleatoriamente
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward propagation"""
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, learning_rate):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        """Entrena la red"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + 
                          (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Hace predicciones"""
        return (self.forward(X) > 0.5).astype(int)
```

## 🔥 PyTorch Básico

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    """Red neuronal simple con PyTorch"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Crear modelo
model = SimpleNN(input_size=10, hidden_size=20, output_size=1)

# Definir loss y optimizador
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## 📊 Funciones de Pérdida

### Binary Cross-Entropy (clasificación binaria)
$$L = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

### Categorical Cross-Entropy (multi-clase)
$$L = -\frac{1}{n}\sum\sum y_i\log(\hat{y}_i)$$

### Mean Squared Error (regresión)
$$L = \frac{1}{n}\sum(y - \hat{y})^2$$

## ⚙️ Optimizadores

### SGD (Stochastic Gradient Descent)
$$W = W - \alpha \nabla J(W)$$

### Momentum
Acelera el descenso en la dirección correcta.

### Adam (Adaptive Moment Estimation)
Combina momentum con learning rate adaptativo.

```python
# En PyTorch
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## 🎯 Ejercicios

### Ejercicio 1: XOR Problem
Implementa una red que resuelva el problema XOR (no linealmente separable).

### Ejercicio 2: MNIST Digits
Clasifica dígitos escritos a mano usando una red neuronal simple.

### Ejercicio 3: Comparación
Compara tu implementación desde cero vs PyTorch.

## 💡 Tips Importantes

### Inicialización de Pesos
```python
# Xavier/Glorot initialization
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

### Batch Normalization
Normaliza las activaciones entre capas.

### Dropout
Apaga neuronas aleatoriamente durante entrenamiento para evitar overfitting.

```python
self.dropout = nn.Dropout(p=0.5)
```

### Learning Rate Scheduling
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

## 🔍 Debugging Tips

```python
# Verifica shapes
print(f"Input shape: {X.shape}")
print(f"Weights shape: {W1.shape}")
print(f"Output shape: {output.shape}")

# Checa gradientes
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Visualiza loss
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

## ✅ Checklist

- [ ] Entiendo el perceptrón
- [ ] Conozco las funciones de activación
- [ ] Puedo implementar forward propagation
- [ ] Entiendo backpropagation
- [ ] Sé usar PyTorch básico
- [ ] Completé el ejercicio XOR
- [ ] Entrené un modelo para MNIST

---

**Siguiente**: CNNs para Computer Vision 🖼️
