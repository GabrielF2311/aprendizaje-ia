# 🎯 Vectores - Fundamentos

## ¿Qué es un vector?

Un **vector** es una cantidad que tiene tanto **magnitud** (tamaño) como **dirección**. En matemáticas e IA, los vectores son listas ordenadas de números.

### Representaciones

**Geométrica**: Imagina una flecha en el espacio
- Tiene un punto inicial (origen)
- Tiene un punto final
- La dirección es hacia donde apunta
- La longitud es su magnitud

**Algebraica**: Una lista de números
```
v = [3, 4]  # Vector 2D
v = [1, 2, 3]  # Vector 3D
v = [x₁, x₂, x₃, ..., xₙ]  # Vector n-dimensional
```

**Notación matemática**:
```
v⃗ = (3, 4)
v⃗ = [3]
    [4]
```

## Dimensión de un Vector

La **dimensión** de un vector es la cantidad de componentes que tiene.

- Vector 2D: `[3, 4]` → 2 componentes (x, y)
- Vector 3D: `[1, 2, 3]` → 3 componentes (x, y, z)
- Vector 100D: `[x₁, x₂, ..., x₁₀₀]` → 100 componentes

### En Machine Learning

En ML, cada **ejemplo** (data point) se representa como un vector:

```python
# Ejemplo: Una casa
casa = [
    150,    # metros cuadrados
    3,      # número de habitaciones
    2,      # número de baños
    2020,   # año de construcción
    500000  # precio
]
# Este es un vector de 5 dimensiones (5 features)
```

## Magnitud (Norma) de un Vector

La **magnitud** o **norma** es la "longitud" del vector.

### Fórmula (Norma L2 / Euclidiana)

$$||v|| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$$

### Ejemplos

**Vector 2D**: `v = [3, 4]`
```
||v|| = √(3² + 4²)
     = √(9 + 16)
     = √25
     = 5
```

**Vector 3D**: `v = [1, 2, 2]`
```
||v|| = √(1² + 2² + 2²)
     = √(1 + 4 + 4)
     = √9
     = 3
```

### En Python

```python
import math

def magnitude(vector):
    """Calcula la magnitud de un vector"""
    sum_of_squares = sum(x**2 for x in vector)
    return math.sqrt(sum_of_squares)

# Uso
v = [3, 4]
print(magnitude(v))  # 5.0
```

## Vector Unitario (Normalización)

Un **vector unitario** es un vector con magnitud = 1.

### ¿Para qué sirve?

- Mantener solo la dirección, eliminar la escala
- Útil en comparaciones de similitud
- Fundamental en redes neuronales (normalización)

### Fórmula de Normalización

$$\hat{v} = \frac{v}{||v||}$$

Divides cada componente por la magnitud.

### Ejemplo

**Vector**: `v = [3, 4]`
**Magnitud**: `||v|| = 5`

**Normalización**:
```
v̂ = [3/5, 4/5]
  = [0.6, 0.8]
```

**Verificación**:
```
||v̂|| = √(0.6² + 0.8²)
     = √(0.36 + 0.64)
     = √1
     = 1 ✓
```

### En Python

```python
def normalize(vector):
    """Normaliza un vector"""
    mag = magnitude(vector)
    if mag == 0:
        raise ValueError("No se puede normalizar el vector cero")
    return [x / mag for x in vector]

# Uso
v = [3, 4]
v_norm = normalize(v)
print(v_norm)  # [0.6, 0.8]
print(magnitude(v_norm))  # 1.0
```

## Distancia entre Vectores

La **distancia euclidiana** mide qué tan "lejos" están dos vectores.

### Fórmula

$$d(v_1, v_2) = ||v_1 - v_2||$$

Es la magnitud del vector diferencia.

### Ejemplo

`v1 = [1, 2]`, `v2 = [4, 6]`

**Paso 1**: Resta componente a componente
```
v1 - v2 = [1-4, 2-6] = [-3, -4]
```

**Paso 2**: Calcula la magnitud
```
d = ||[-3, -4]||
  = √((-3)² + (-4)²)
  = √(9 + 16)
  = √25
  = 5
```

### En ML: K-Nearest Neighbors

El algoritmo KNN usa distancia euclidiana para encontrar ejemplos similares:

```python
def distance(v1, v2):
    """Calcula distancia euclidiana entre dos vectores"""
    if len(v1) != len(v2):
        raise ValueError("Vectores deben tener la misma dimensión")
    
    diff = [a - b for a, b in zip(v1, v2)]
    return magnitude(diff)

# Ejemplo: ¿Qué casa es más similar?
casa_referencia = [150, 3, 2, 2020]
casa_a = [160, 3, 2, 2019]
casa_b = [200, 4, 3, 2015]

d_a = distance(casa_referencia, casa_a)
d_b = distance(casa_referencia, casa_b)

print(f"Distancia a casa A: {d_a}")  # Más similar
print(f"Distancia a casa B: {d_b}")  # Menos similar
```

## Vectores Especiales

### Vector Cero

Todos sus componentes son 0.
```
0⃗ = [0, 0, 0]
```

**Propiedades**:
- Magnitud = 0
- No tiene dirección definida
- Es el elemento neutro de la suma

### Vectores de la Base Estándar

Vectores con 1 en una posición y 0 en el resto.

**En 2D**:
```
e₁ = [1, 0]  # Eje X
e₂ = [0, 1]  # Eje Y
```

**En 3D**:
```
e₁ = [1, 0, 0]  # Eje X
e₂ = [0, 1, 0]  # Eje Y
e₃ = [0, 0, 1]  # Eje Z
```

**Importancia**: Cualquier vector se puede escribir como combinación de vectores base.

```
v = [3, 4] = 3·e₁ + 4·e₂ = 3·[1,0] + 4·[0,1]
```

## Visualización en Python

```python
import matplotlib.pyplot as plt

def plot_vector_2d(vector, origin=[0, 0], color='blue', label=''):
    """Dibuja un vector 2D"""
    plt.quiver(origin[0], origin[1], 
               vector[0], vector[1],
               angles='xy', scale_units='xy', scale=1,
               color=color, label=label)
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')

# Ejemplo: Dibujar vector [3, 4]
v = [3, 4]
plot_vector_2d(v, label=f'v = {v}')
plt.title(f'Vector v = {v}\nMagnitud: {magnitude(v):.2f}')
plt.show()
```

## Conexión con IA

### ¿Por qué vectores en IA?

1. **Representación de datos**: Cada ejemplo es un vector de features
2. **Similitud**: Distancia entre vectores mide similitud
3. **Embeddings**: Palabras, imágenes → vectores densos
4. **Operaciones eficientes**: Álgebra lineal es muy rápida

### Ejemplos en IA

**Word Embeddings**:
```
"rey" = [0.2, 0.5, 0.1, ...]  # Vector de 300 dimensiones
"reina" = [0.25, 0.48, 0.12, ...]
```

**Imagen**:
```
Una imagen 28x28 = vector de 784 dimensiones
(cada píxel es una componente)
```

**Features de ML**:
```
cliente = [
    edad,
    salario,
    años_como_cliente,
    compras_mensuales,
    ...
]
```

## Ejercicios Conceptuales

1. **Pregunta**: ¿Qué vector tiene mayor magnitud: [3, 4] o [1, 1, 1, 1, 1]?
   <details>
   <summary>Respuesta</summary>
   [3, 4] → ||v|| = 5
   [1, 1, 1, 1, 1] → ||v|| = √5 ≈ 2.24
   
   [3, 4] tiene mayor magnitud.
   </details>

2. **Pregunta**: ¿Puedes normalizar el vector [0, 0, 0]?
   <details>
   <summary>Respuesta</summary>
   No, porque dividirías por 0 (su magnitud es 0).
   El vector cero no tiene dirección definida.
   </details>

3. **Pregunta**: Si dos vectores tienen la misma dirección pero diferentes magnitudes, ¿serán iguales después de normalizarlos?
   <details>
   <summary>Respuesta</summary>
   Sí, la normalización elimina la escala y mantiene solo la dirección.
   </details>

## Resumen

| Concepto | Fórmula | Interpretación |
|----------|---------|----------------|
| **Vector** | `v = [v₁, v₂, ..., vₙ]` | Lista ordenada de números |
| **Dimensión** | n | Número de componentes |
| **Magnitud** | `‖v‖ = √(v₁² + v₂² + ... + vₙ²)` | "Longitud" del vector |
| **Normalización** | `v̂ = v / ‖v‖` | Vector con magnitud 1 |
| **Distancia** | `d(u,v) = ‖u - v‖` | Qué tan lejos están dos vectores |

## Siguiente Paso

Mañana aprenderás **operaciones con vectores**:
- Suma y resta
- Multiplicación por escalar
- Producto punto (dot product)
- Ángulos entre vectores

¡Completa los ejercicios de hoy antes de avanzar! 🚀
