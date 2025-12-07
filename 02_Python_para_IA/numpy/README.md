# 🔢 NumPy Mastery - Semana 11

## 🎯 Objetivos de la Semana

- Dominar arrays de NumPy
- Operaciones vectorizadas (100x más rápidas que loops)
- Broadcasting y manipulación de formas
- Álgebra lineal con NumPy
- Implementar red neuronal desde cero

## 📚 Por qué NumPy es Crucial para IA

NumPy es la **base de todo** en Python científico:
- PyTorch y TensorFlow usan conceptos similares
- Operaciones matriciales son fundamentales en ML/DL
- 50-100x más rápido que Python puro
- Sintaxis similar a MATLAB/Julia

---

## 📅 Plan de la Semana

### **Día 1: Arrays Básicos**
- ¿Qué es un array?
- Creación de arrays
- Atributos (shape, dtype, ndim)
- Arrays especiales (zeros, ones, arange)

📖 **Teoría**: `teoria/01_arrays_basicos.md`
💻 **Ejercicios**: `ejercicios/dia_01_arrays.py`

---

### **Día 2: Indexing y Slicing**
- Indexación básica vs avanzada
- Slicing multidimensional
- Boolean indexing
- Fancy indexing

📖 **Teoría**: `teoria/02_indexing_slicing.md`
💻 **Ejercicios**: `ejercicios/dia_02_operaciones.py`

---

### **Día 3: Broadcasting y Operaciones**
- ¿Qué es broadcasting?
- Reglas de broadcasting
- Operaciones elemento a elemento
- Funciones universales (ufuncs)

📖 **Teoría**: `teoria/03_broadcasting.md`
💻 **Ejercicios**: `ejercicios/dia_03_broadcasting.py`

---

### **Día 4-5: Álgebra Lineal**
- Multiplicación de matrices
- Transposición
- Inversas y determinantes
- Eigenvalues/eigenvectors
- Descomposición SVD

📖 **Teoría**: `teoria/04_algebra_lineal.md`
💻 **Ejercicios**: `ejercicios/dia_04_algebra.py`

---

### **Día 6-7: PROYECTO - Red Neuronal con NumPy**

Implementa una red neuronal completamente funcional usando solo NumPy.

**Objetivos**:
- Forward propagation
- Backward propagation
- Gradient descent
- Entrenar en MNIST simplificado

💻 **Código**: `proyecto_numpy.py`

---

## 🔑 Conceptos Clave

### Array vs Lista

```python
# Lista de Python
lista = [1, 2, 3, 4]
lista2 = [x * 2 for x in lista]  # Loop implícito

# NumPy array
import numpy as np
arr = np.array([1, 2, 3, 4])
arr2 = arr * 2  # Vectorizado, súper rápido!
```

### Ventajas de NumPy

✅ **Velocidad**: Implementado en C
✅ **Memoria**: Más eficiente
✅ **Sintaxis**: Más limpia y expresiva
✅ **Broadcasting**: Operaciones automáticas entre shapes compatibles

---

## 📊 Comparación de Rendimiento

```python
import numpy as np
import time

# Python puro
lista = list(range(1000000))
start = time.time()
resultado = [x * 2 for x in lista]
print(f"Python: {time.time() - start:.4f}s")

# NumPy
arr = np.arange(1000000)
start = time.time()
resultado = arr * 2
print(f"NumPy: {time.time() - start:.4f}s")

# NumPy es ~50x más rápido!
```

---

## ✅ Checklist de Progreso

### Conceptos Fundamentales
- [ ] Entiendo la diferencia entre array y lista
- [ ] Puedo crear arrays de diferentes formas
- [ ] Sé usar shape, dtype, ndim
- [ ] Entiendo el concepto de axis

### Operaciones
- [ ] Domino slicing multidimensional
- [ ] Uso boolean indexing correctamente
- [ ] Aplico broadcasting
- [ ] Conozco las ufuncs principales

### Álgebra Lineal
- [ ] Multiplico matrices correctamente
- [ ] Uso transpose
- [ ] Calculo inversas y determinantes
- [ ] Aplico SVD

### Proyecto
- [ ] Implementé forward propagation
- [ ] Implementé backward propagation
- [ ] Entrené un modelo funcional
- [ ] Logré >80% accuracy

---

## 🎯 Mini-Desafíos Diarios

**Día 1**: Crea un array 3D y visualiza su estructura
**Día 2**: Extrae elementos de una imagen (array 2D) usando slicing avanzado
**Día 3**: Normaliza un dataset sin usar loops
**Día 4**: Implementa multiplicación de matrices sin usar `@` o `dot`
**Día 5**: Calcula PCA manualmente con NumPy
**Día 6-7**: Red neuronal funcionando

---

## 📚 Recursos

### Documentación
- [NumPy Documentation](https://numpy.org/doc/)
- [NumPy for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)

### Videos
- **freeCodeCamp** - NumPy Tutorial (YouTube)
- **Keith Galli** - Complete NumPy Tutorial

### Práctica
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)

---

## 💡 Tips para Esta Semana

1. **Piensa en vectores/matrices**, no en loops
2. **Verifica shapes constantemente**: `print(arr.shape)`
3. **Experimenta en Jupyter**: Prueba cada operación
4. **Compara con Python puro**: Aprecia la velocidad
5. **Lee errores cuidadosamente**: NumPy da buenos mensajes

---

## 🔗 Conexión con Deep Learning

```python
# Lo que harás con NumPy esta semana...
z = np.dot(W, x) + b  # Forward pass
dW = np.dot(x, dz.T)  # Backward pass

# ...es exactamente lo que hace PyTorch internamente!
z = torch.matmul(W, x) + b
```

---

## 🚀 Siguiente Paso

Empieza con **Día 1**: Lee `teoria/01_arrays_basicos.md` y completa `ejercicios/dia_01_arrays.py`

**¡Que tengas una excelente semana con NumPy!** 🔢
