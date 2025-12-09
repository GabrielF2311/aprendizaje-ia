# 🎯 Operaciones con Vectores

## Producto Punto (Dot Product)

El **producto punto** es una de las operaciones más importantes en álgebra lineal y machine learning.

### Definición

$$\vec{v_1} \cdot \vec{v_2} = v_{1,1} \cdot v_{2,1} + v_{1,2} \cdot v_{2,2} + ... + v_{1,n} \cdot v_{2,n}$$

O en forma compacta:
$$\vec{v_1} \cdot \vec{v_2} = \sum_{i=1}^{n} v_{1,i} \cdot v_{2,i}$$

### Ejemplo

```
v1 = [1, 2, 3]
v2 = [4, 5, 6]

v1 · v2 = (1×4) + (2×5) + (3×6)
        = 4 + 10 + 18
        = 32
```

### Propiedades

1. **Conmutativo**: $\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a}$
2. **Distributivo**: $\vec{a} \cdot (\vec{b} + \vec{c}) = \vec{a} \cdot \vec{b} + \vec{a} \cdot \vec{c}$
3. **Asociativo con escalar**: $(c\vec{a}) \cdot \vec{b} = c(\vec{a} \cdot \vec{b})$

### En Python

```python
# Manual
def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

# NumPy
import numpy as np
result = np.dot(v1, v2)
# o
result = v1 @ v2  # Operador @
```

---

## Magnitud con Producto Punto

La magnitud de un vector es su producto punto consigo mismo:

$$||\vec{v}|| = \sqrt{\vec{v} \cdot \vec{v}}$$

```python
magnitude = math.sqrt(dot_product(v, v))
# O más directo:
magnitude = np.linalg.norm(v)
```

---

## Ángulo entre Vectores

### Fórmula

$$\cos(\theta) = \frac{\vec{v_1} \cdot \vec{v_2}}{||\vec{v_1}|| \cdot ||\vec{v_2}||}$$

Despejando θ:
$$\theta = \arccos\left(\frac{\vec{v_1} \cdot \vec{v_2}}{||\vec{v_1}|| \cdot ||\vec{v_2}||}\right)$$

### Casos Especiales

| θ | cos(θ) | v1 · v2 | Interpretación |
|---|--------|---------|----------------|
| 0° | 1 | máximo positivo | Misma dirección |
| 90° | 0 | 0 | Perpendiculares |
| 180° | -1 | máximo negativo | Direcciones opuestas |

### Ejemplo

```python
import math

def angle_between(v1, v2, degrees=True):
    dot = dot_product(v1, v2)
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    
    cos_theta = dot / (mag1 * mag2)
    theta_rad = math.acos(cos_theta)
    
    if degrees:
        return math.degrees(theta_rad)
    return theta_rad

# Ejemplo
v1 = [1, 0]
v2 = [0, 1]
angle = angle_between(v1, v2)  # 90°
```

---

## Vectores Perpendiculares (Ortogonales)

Dos vectores son **perpendiculares** si su producto punto es cero.

$$\vec{v_1} \perp \vec{v_2} \iff \vec{v_1} \cdot \vec{v_2} = 0$$

### Ejemplos

```python
# Perpendiculares
[1, 0] · [0, 1] = 0  ✓
[3, 4] · [-4, 3] = -12 + 12 = 0  ✓

# No perpendiculares
[1, 1] · [1, 1] = 2  ✗
```

### Base Ortonormal

Un conjunto de vectores es **ortonormal** si:
1. Todos son perpendiculares entre sí (ortogonales)
2. Todos tienen magnitud 1 (unitarios)

```python
# Base estándar en 3D (ortonormal)
e1 = [1, 0, 0]
e2 = [0, 1, 0]
e3 = [0, 0, 1]

# Verifica:
e1 · e2 = 0  # Ortogonales
||e1|| = 1   # Unitarios
```

---

## Proyección de Vectores

La **proyección** de $\vec{v}$ sobre $\vec{u}$ es la "sombra" de $\vec{v}$ en la dirección de $\vec{u}$.

### Fórmula

$$\text{proj}_{\vec{u}}(\vec{v}) = \frac{\vec{v} \cdot \vec{u}}{\vec{u} \cdot \vec{u}} \vec{u}$$

Si $\vec{u}$ es unitario (||u|| = 1):
$$\text{proj}_{\vec{u}}(\vec{v}) = (\vec{v} \cdot \vec{u}) \vec{u}$$

### Visualización

```
v
|
|    /
|   / proj_u(v)
|  /
| /
|/_________ u
```

### Ejemplo

```python
def project_onto(v, u):
    """Proyecta v sobre u"""
    scalar = dot_product(v, u) / dot_product(u, u)
    return [scalar * ui for ui in u]

# Proyectar [3, 4] sobre el eje X [1, 0]
v = [3, 4]
u = [1, 0]
proj = project_onto(v, u)  # [3, 0]
```

### Componentes Paralela y Perpendicular

Cualquier vector se puede descomponer en:

$$\vec{v} = \vec{v}_{\parallel} + \vec{v}_{\perp}$$

Donde:
- $\vec{v}_{\parallel}$ = proyección sobre $\vec{u}$
- $\vec{v}_{\perp} = \vec{v} - \vec{v}_{\parallel}$

```python
v_parallel = project_onto(v, u)
v_perpendicular = [v[i] - v_parallel[i] for i in range(len(v))]
```

---

## Producto Cruz (Cross Product) - Solo 3D

El producto cruz produce un vector **perpendicular** a ambos vectores de entrada.

### Fórmula

$$\vec{a} \times \vec{b} = \begin{bmatrix} 
a_2b_3 - a_3b_2 \\
a_3b_1 - a_1b_3 \\
a_1b_2 - a_2b_1
\end{bmatrix}$$

### Método del Determinante

$$\vec{a} \times \vec{b} = \begin{vmatrix}
\hat{i} & \hat{j} & \hat{k} \\
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{vmatrix}$$

### Ejemplo

```python
a = [1, 0, 0]  # Eje X
b = [0, 1, 0]  # Eje Y

a × b = [0*0 - 0*1,    # 0
         0*0 - 1*0,    # 0
         1*1 - 0*0]    # 1
      = [0, 0, 1]  # Eje Z!
```

### Propiedades

1. **No conmutativo**: $\vec{a} \times \vec{b} = -(\vec{b} \times \vec{a})$
2. **Magnitud**: $||\vec{a} \times \vec{b}|| = ||\vec{a}|| \cdot ||\vec{b}|| \cdot \sin(\theta)$
3. **Perpendicular**: $(\vec{a} \times \vec{b}) \cdot \vec{a} = 0$ y $(\vec{a} \times \vec{b}) \cdot \vec{b} = 0$

### Regla de la Mano Derecha

```
    Z (arriba)
    |
    |
    |_____ Y
   /
  /
 X

X × Y = Z
Y × Z = X
Z × X = Y
```

### Aplicaciones

- **Física**: Torque, momento angular
- **Geometría**: Normal a un plano (gráficos 3D)
- **ML**: Menos común, pero útil en geometría computacional

---

## Combinaciones Lineales

Una **combinación lineal** es:

$$c_1\vec{v_1} + c_2\vec{v_2} + ... + c_n\vec{v_n}$$

### Ejemplo

```python
# Cualquier vector en 2D se puede expresar como:
v = c1 * [1, 0] + c2 * [0, 1]

# Por ejemplo:
[3, 4] = 3 * [1, 0] + 4 * [0, 1]
```

### Espacio Generado (Span)

El **span** de un conjunto de vectores es el conjunto de todas sus combinaciones lineales posibles.

```python
# Span de [1, 0] y [0, 1] es todo R²
# Cualquier punto (x, y) se puede alcanzar
```

---

## Aplicaciones en Machine Learning

### 1. Similitud de Documentos

```python
# Dos documentos representados como vectores
doc1 = [3, 1, 0, 2]  # Frecuencia de palabras
doc2 = [2, 0, 1, 1]

# Similitud = coseno del ángulo
similarity = dot_product(doc1, doc2) / (magnitude(doc1) * magnitude(doc2))
```

### 2. Redes Neuronales

```python
# Forward pass en una neurona
weights = [0.5, 0.3, 0.2]
inputs = [1.0, 2.0, 3.0]

# Salida = producto punto + bias
output = dot_product(weights, inputs) + bias
```

### 3. Regresión Lineal

```python
# Predicción: ŷ = w · x + b
y_pred = dot_product(weights, features) + bias
```

---

## Ejercicios Conceptuales

### 1. ¿Qué significa un producto punto negativo?
<details>
<summary>Respuesta</summary>
El ángulo entre los vectores es mayor a 90° (entre 90° y 180°). Los vectores "apuntan" en direcciones generalmente opuestas.
</details>

### 2. ¿Cuándo es útil normalizar vectores antes de calcular el producto punto?
<details>
<summary>Respuesta</summary>
Cuando solo nos interesa la dirección, no la magnitud. El producto punto de vectores normalizados es exactamente cos(θ), una medida directa de similitud.
</details>

### 3. ¿Por qué el producto cruz solo existe en 3D?
<details>
<summary>Respuesta</summary>
En 2D no hay una dirección "perpendicular" única. En 4D+ hay múltiples direcciones perpendiculares. Solo en 3D hay exactamente una dirección perpendicular única (usando la regla de la mano derecha).
</details>

---

## Resumen de Fórmulas

| Operación | Fórmula | Resultado |
|-----------|---------|-----------|
| **Producto Punto** | $\vec{a} \cdot \vec{b} = \sum a_ib_i$ | Escalar |
| **Ángulo** | $\theta = \arccos\left(\frac{\vec{a} \cdot \vec{b}}{\\|\vec{a}\\| \\|\vec{b}\\|}\right)$ | Ángulo |
| **Proyección** | $\text{proj}_{\vec{b}}(\vec{a}) = \frac{\vec{a} \cdot \vec{b}}{\\|\vec{b}\\|^2}\vec{b}$ | Vector |
| **Producto Cruz** | $\vec{a} \times \vec{b}$ | Vector ⊥ |

---

**Siguiente**: Día 3 - Matrices y operaciones matriciales 🔢
