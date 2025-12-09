# Día 7: Transformaciones Lineales y Geométricas

## 📋 Objetivos del Día
- Comprender qué es una transformación lineal
- Dominar matrices de transformación (rotación, escalado, traslación)
- Aplicar transformaciones en 2D y 3D
- Implementar transformaciones compuestas
- Reconocer aplicaciones en Computer Vision, Gráficos y Deep Learning

---

## 1. Fundamentos de Transformaciones Lineales

### 1.1 Definición

Una **transformación lineal** es una función $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ que satisface:

1. **Aditividad:** $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneidad:** $T(c\mathbf{u}) = cT(\mathbf{u})$

**Propiedad clave:** Toda transformación lineal se puede representar como una multiplicación matricial:

$$
T(\mathbf{x}) = A\mathbf{x}
$$

Donde **A** es la **matriz de transformación**.

### 1.2 Propiedades Importantes

- El origen siempre se mapea a sí mismo: $T(\mathbf{0}) = \mathbf{0}$
- Las líneas rectas se mantienen rectas
- Las líneas paralelas permanecen paralelas
- El origen del sistema de coordenadas no cambia

### 1.3 Ejemplo Simple

**Escalado por 2:**
$$
T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 2x \\ 2y \end{pmatrix}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Punto original
v = np.array([1, 2])

# Matriz de escalado
A = np.array([[2, 0],
              [0, 2]])

# Aplicar transformación
v_transformed = A @ v

print(f"Original: {v}")       # [1, 2]
print(f"Transformado: {v_transformed}")  # [2, 4]
```

---

## 2. Transformaciones Básicas en 2D

### 2.1 Escalado (Scaling)

**Escalado uniforme** (mismo factor en todas direcciones):
$$
S(s) = \begin{bmatrix} s & 0 \\ 0 & s \end{bmatrix}
$$

**Escalado no uniforme** (diferentes factores):
$$
S(s_x, s_y) = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}
$$

**Ejemplo:**
```python
import numpy as np

# Escalar 2× en X, 3× en Y
S = np.array([[2, 0],
              [0, 3]])

punto = np.array([1, 1])
resultado = S @ punto

print(resultado)  # [2, 3]
```

**Visualización:**
```python
import matplotlib.pyplot as plt

# Cuadrado original
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# Aplicar escalado
S = np.array([[2, 0],
              [0, 0.5]])
square_scaled = S @ square

# Graficar
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(square[0], square[1], 'b-o')
plt.title('Original')
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(square_scaled[0], square_scaled[1], 'r-o')
plt.title('Escalado (2×, 0.5×)')
plt.grid(True)
plt.axis('equal')
plt.show()
```

### 2.2 Reflexión (Reflection)

**Reflexión sobre el eje X:**
$$
R_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

**Reflexión sobre el eje Y:**
$$
R_y = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}
$$

**Reflexión sobre el origen:**
$$
R_o = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}
$$

**Reflexión sobre la línea y = x:**
$$
R_{y=x} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

**Ejemplo:**
```python
# Reflexión sobre eje X
R_x = np.array([[1, 0],
                [0, -1]])

punto = np.array([2, 3])
reflejado = R_x @ punto

print(f"Original: {punto}")      # [2, 3]
print(f"Reflejado: {reflejado}") # [2, -3]
```

### 2.3 Rotación (Rotation)

**Rotación antihoraria por ángulo θ:**
$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

**Derivación intuitiva:**
- El vector $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ rota a $\begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix}$
- El vector $\begin{pmatrix} 0 \\ 1 \end{pmatrix}$ rota a $\begin{pmatrix} -\sin\theta \\ \cos\theta \end{pmatrix}$

**Rotaciones comunes:**

**90° antihorario:**
$$
R(90°) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

**180°:**
$$
R(180°) = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}
$$

**Implementación:**
```python
def rotation_matrix_2d(theta_degrees):
    """Crea matriz de rotación 2D"""
    theta = np.radians(theta_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# Rotar 45 grados
R = rotation_matrix_2d(45)
punto = np.array([1, 0])
rotado = R @ punto

print(f"Original: {punto}")
print(f"Rotado 45°: {rotado}")
# [0.707, 0.707] ≈ [√2/2, √2/2]
```

**Visualización de rotación:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Vector original
v = np.array([3, 1])

# Rotar en incrementos de 30°
angles = np.arange(0, 360, 30)

plt.figure(figsize=(8, 8))
for angle in angles:
    R = rotation_matrix_2d(angle)
    v_rot = R @ v
    plt.arrow(0, 0, v_rot[0], v_rot[1], 
              head_width=0.2, head_length=0.3, 
              fc='blue', ec='blue', alpha=0.5)

plt.arrow(0, 0, v[0], v[1], 
          head_width=0.2, head_length=0.3, 
          fc='red', ec='red', linewidth=2, label='Original')

plt.grid(True)
plt.axis('equal')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend()
plt.title('Rotaciones cada 30°')
plt.show()
```

### 2.4 Cizallamiento (Shear)

**Cizallamiento horizontal:**
$$
H_x(k) = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}
$$

**Cizallamiento vertical:**
$$
H_y(k) = \begin{bmatrix} 1 & 0 \\ k & 1 \end{bmatrix}
$$

**Ejemplo:**
```python
# Cizallamiento horizontal
H = np.array([[1, 0.5],
              [0, 1]])

# Cuadrado
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

sheared = H @ square

# Visualizar
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(square[0], square[1], 'b-o')
plt.title('Cuadrado Original')
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(sheared[0], sheared[1], 'r-o')
plt.title('Después de Cizallamiento')
plt.grid(True)
plt.axis('equal')
plt.show()
```

---

## 3. Transformaciones en 3D

### 3.1 Rotación 3D

**Rotación alrededor del eje X:**
$$
R_x(\theta) = \begin{bmatrix} 
1 & 0 & 0 \\
0 & \cos\theta & -\sin\theta \\
0 & \sin\theta & \cos\theta
\end{bmatrix}
$$

**Rotación alrededor del eje Y:**
$$
R_y(\theta) = \begin{bmatrix} 
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix}
$$

**Rotación alrededor del eje Z:**
$$
R_z(\theta) = \begin{bmatrix} 
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

**Implementación:**
```python
def rotation_matrix_3d(axis, theta_degrees):
    """
    Crea matriz de rotación 3D
    axis: 'x', 'y', o 'z'
    """
    theta = np.radians(theta_degrees)
    c, s = np.cos(theta), np.sin(theta)
    
    if axis == 'x':
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

# Ejemplo: Rotar punto alrededor del eje Z
punto_3d = np.array([1, 0, 0])
R_z = rotation_matrix_3d('z', 90)
rotado_3d = R_z @ punto_3d

print(f"Original: {punto_3d}")
print(f"Rotado 90° (eje Z): {rotado_3d}")
# [0, 1, 0] aproximadamente
```

### 3.2 Escalado 3D

$$
S(s_x, s_y, s_z) = \begin{bmatrix} 
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & s_z
\end{bmatrix}
$$

### 3.3 Reflexión 3D

**Reflexión sobre plano XY (z = 0):**
$$
R_{xy} = \begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & -1
\end{bmatrix}
$$

---

## 4. Coordenadas Homogéneas y Traslación

### 4.1 El Problema de la Traslación

La traslación **NO** es una transformación lineal:
$$
T(\mathbf{x}) = \mathbf{x} + \mathbf{t}
$$

No se puede representar como $A\mathbf{x}$ solamente.

### 4.2 Coordenadas Homogéneas

Agregar una dimensión extra (coordenada homogénea = 1):

**2D:**
$$
\begin{pmatrix} x \\ y \end{pmatrix} \rightarrow \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}
$$

**Traslación en coordenadas homogéneas:**
$$
\begin{bmatrix} 
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{bmatrix}
\begin{pmatrix} x \\ y \\ 1 \end{pmatrix}
=
\begin{pmatrix} x + t_x \\ y + t_y \\ 1 \end{pmatrix}
$$

**Ejemplo:**
```python
def translation_matrix_2d(tx, ty):
    """Matriz de traslación en coordenadas homogéneas"""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

# Trasladar punto (2, 3) por (5, -2)
punto_h = np.array([2, 3, 1])  # Coordenadas homogéneas
T = translation_matrix_2d(5, -2)

resultado = T @ punto_h
print(resultado)  # [7, 1, 1]

# Convertir de vuelta a coordenadas cartesianas
punto_final = resultado[:2] / resultado[2]
print(punto_final)  # [7, 1]
```

### 4.3 Transformaciones Combinadas en Coordenadas Homogéneas

**Rotación:**
$$
R(\theta) = \begin{bmatrix} 
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

**Escalado:**
$$
S(s_x, s_y) = \begin{bmatrix} 
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

**Traslación:**
$$
T(t_x, t_y) = \begin{bmatrix} 
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{bmatrix}
$$

---

## 5. Composición de Transformaciones

### 5.1 Multiplicación de Matrices

Para aplicar múltiples transformaciones, multiplicar las matrices:

$$
T_{\text{total}} = T_3 \cdot T_2 \cdot T_1
$$

⚠️ **Orden importa:** Las transformaciones se aplican de **derecha a izquierda**.

### 5.2 Ejemplo: Rotar alrededor de un punto

**Problema:** Rotar un objeto 90° alrededor del punto (3, 2)

**Pasos:**
1. Trasladar para que el centro de rotación esté en el origen: $T(-3, -2)$
2. Rotar: $R(90°)$
3. Trasladar de vuelta: $T(3, 2)$

$$
M = T(3, 2) \cdot R(90°) \cdot T(-3, -2)
$$

**Implementación:**
```python
def rotation_matrix_2d_homogeneous(theta_degrees):
    """Matriz de rotación en coordenadas homogéneas"""
    theta = np.radians(theta_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

# Transformaciones
T1 = translation_matrix_2d(-3, -2)
R = rotation_matrix_2d_homogeneous(90)
T2 = translation_matrix_2d(3, 2)

# Composición (orden: T2 @ R @ T1)
M = T2 @ R @ T1

# Aplicar a un punto
punto = np.array([5, 2, 1])
resultado = M @ punto

print(f"Punto original: {punto[:2]}")
print(f"Después de rotar 90° alrededor de (3,2): {resultado[:2]}")
```

### 5.3 Ejemplo Completo: Pipeline de Transformaciones

```python
import numpy as np
import matplotlib.pyplot as plt

def crear_triangulo():
    """Crea un triángulo simple"""
    return np.array([[0, 2, 1, 0],    # X coords
                     [0, 0, 2, 0],    # Y coords
                     [1, 1, 1, 1]])   # Homogeneous coords

# Triángulo original
triangulo = crear_triangulo()

# Pipeline de transformaciones:
# 1. Escalar 1.5×
S = np.array([[1.5, 0, 0],
              [0, 1.5, 0],
              [0, 0, 1]])

# 2. Rotar 45°
R = rotation_matrix_2d_homogeneous(45)

# 3. Trasladar (4, 3)
T = translation_matrix_2d(4, 3)

# Combinar transformaciones
M_total = T @ R @ S

# Aplicar
triangulo_transformado = M_total @ triangulo

# Visualizar
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(triangulo[0], triangulo[1], 'b-o', linewidth=2, label='Original')
plt.title('Triángulo Original')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(triangulo_transformado[0], triangulo_transformado[1], 
         'r-o', linewidth=2, label='Transformado')
plt.title('Después: Escalar → Rotar → Trasladar')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 6. Aplicaciones en Computer Vision

### 6.1 Data Augmentation

Aumentar dataset de imágenes aplicando transformaciones aleatorias:

```python
def augment_image_transforms(image_points):
    """
    Aplica transformaciones aleatorias para data augmentation
    image_points: Coordenadas de la imagen en formato homogéneo
    """
    # Rotación aleatoria (-15° a 15°)
    angle = np.random.uniform(-15, 15)
    R = rotation_matrix_2d_homogeneous(angle)
    
    # Escalado aleatorio (0.9× a 1.1×)
    scale = np.random.uniform(0.9, 1.1)
    S = np.array([[scale, 0, 0],
                  [0, scale, 0],
                  [0, 0, 1]])
    
    # Traslación aleatoria
    tx = np.random.uniform(-10, 10)
    ty = np.random.uniform(-10, 10)
    T = translation_matrix_2d(tx, ty)
    
    # Combinar
    M = T @ R @ S
    
    # Aplicar
    return M @ image_points

# Ejemplo con puntos de una imagen
image_corners = np.array([[0, 100, 100, 0, 0],
                          [0, 0, 100, 100, 0],
                          [1, 1, 1, 1, 1]])

augmented = augment_image_transforms(image_corners)
```

### 6.2 Corrección de Perspectiva

Transformación de perspectiva (homografía) en visión:

```python
# Transformación de perspectiva simple
def perspective_transform():
    """
    Ejemplo de matriz de perspectiva 3×3
    """
    return np.array([[1.2, 0.1, 10],
                     [0.05, 1.1, 5],
                     [0.002, 0.001, 1]])

H = perspective_transform()

# Aplicar a puntos de imagen
punto = np.array([50, 50, 1])
transformado = H @ punto

# Normalizar (dividir por componente homogénea)
punto_final = transformado[:2] / transformado[2]
print(f"Original: {punto[:2]}")
print(f"Transformado: {punto_final}")
```

### 6.3 Detección de Características y Matching

Normalizar puntos antes de calcular transformaciones:

```python
def normalize_points(points):
    """
    Normaliza puntos para estabilidad numérica
    points: (N, 2) array de coordenadas
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    avg_dist = np.mean(np.linalg.norm(points_centered, axis=1))
    scale = np.sqrt(2) / avg_dist
    
    # Matriz de normalización
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    
    return T

# Ejemplo
points = np.array([[100, 200],
                   [150, 250],
                   [120, 210]])

T_norm = normalize_points(points)
print("Matriz de normalización:")
print(T_norm)
```

---

## 7. Aplicaciones en Deep Learning

### 7.1 Spatial Transformer Networks

Aprender transformaciones de forma diferenciable:

```python
def affine_transform_matrix(theta):
    """
    Crea matriz de transformación afín 2×3 para STN
    theta: [6] parámetros [a, b, c, d, e, f]
    Matriz: [[a, b, c],
             [d, e, f]]
    """
    return theta.reshape(2, 3)

# Ejemplo: Identidad (sin transformación)
theta_identity = np.array([1, 0, 0,
                           0, 1, 0])

A = affine_transform_matrix(theta_identity)
print("Transformación identidad:")
print(A)
# [[1, 0, 0],
#  [0, 1, 0]]

# Ejemplo: Rotación + traslación
angle = np.radians(30)
c, s = np.cos(angle), np.sin(angle)
theta_rot = np.array([c, -s, 0.5,
                      s, c, -0.3])

A_rot = affine_transform_matrix(theta_rot)
print("\nRotación 30° + traslación:")
print(A_rot)
```

### 7.2 Convolutional Layer como Transformación

Una capa convolucional puede verse como una transformación lineal:

$$
\text{output} = W * \text{input}
$$

Donde $W$ son los filtros convolucionales.

### 7.3 Normalización Batch

Transformación afín después de normalizar:

$$
y = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

```python
def batch_norm_transform(x, gamma, beta, mean, std):
    """
    Batch normalization como transformación afín
    """
    x_normalized = (x - mean) / std
    return gamma * x_normalized + beta

# Ejemplo
x = np.array([1, 2, 3, 4, 5])
mean = np.mean(x)
std = np.std(x)
gamma = 2.0  # Escala aprendida
beta = 1.0   # Desplazamiento aprendido

y = batch_norm_transform(x, gamma, beta, mean, std)
print(f"Original: {x}")
print(f"Normalizado: {y}")
```

---

## 8. Transformaciones en Gráficos 3D

### 8.1 Pipeline de Renderizado

**Orden típico:**
1. **Model Transform:** Posicionar objeto en el mundo
2. **View Transform:** Cámara/observador
3. **Projection Transform:** 3D → 2D
4. **Viewport Transform:** Pantalla

### 8.2 Matriz de Vista (Look-At)

Posicionar la cámara mirando hacia un objetivo:

```python
def look_at(eye, target, up):
    """
    Crea matriz de vista (view matrix)
    eye: posición de la cámara
    target: punto hacia donde mira
    up: vector "arriba"
    """
    # Forward (z apunta hacia la cámara)
    f = eye - target
    f = f / np.linalg.norm(f)
    
    # Right (x)
    r = np.cross(up, f)
    r = r / np.linalg.norm(r)
    
    # Up (y)
    u = np.cross(f, r)
    
    # Matriz de vista 4×4
    M = np.eye(4)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = f
    M[:3, 3] = -np.array([np.dot(r, eye), 
                          np.dot(u, eye), 
                          np.dot(f, eye)])
    
    return M

# Ejemplo
eye = np.array([0, 0, 5])
target = np.array([0, 0, 0])
up = np.array([0, 1, 0])

view_matrix = look_at(eye, target, up)
print("View Matrix:")
print(view_matrix)
```

### 8.3 Proyección Perspectiva

```python
def perspective_projection(fov, aspect, near, far):
    """
    Crea matriz de proyección perspectiva
    fov: campo de visión (grados)
    aspect: ratio ancho/alto
    near, far: planos de recorte
    """
    fov_rad = np.radians(fov)
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1
    
    return M

# Ejemplo
proj = perspective_projection(fov=60, aspect=16/9, near=0.1, far=100)
print("Projection Matrix:")
print(proj)
```

---

## 9. Invarianzas y Propiedades

### 9.1 Determinante y Área/Volumen

El determinante de una matriz de transformación indica cómo cambia el área/volumen:

$$
\text{Área}_{\text{transformada}} = |\det(A)| \times \text{Área}_{\text{original}}
$$

```python
# Escalado 2× en X, 3× en Y
S = np.array([[2, 0],
              [0, 3]])

det_S = np.linalg.det(S)  # 6

# Un cuadrado 1×1 se transforma en rectángulo 2×3
area_original = 1
area_transformada = abs(det_S) * area_original
print(f"Área transformada: {area_transformada}")  # 6
```

### 9.2 Transformaciones Ortogonales

Una matriz es **ortogonal** si $Q^T Q = I$

**Propiedades:**
- Preservan longitudes: $\|Q\mathbf{x}\| = \|\mathbf{x}\|$
- Preservan ángulos
- $\det(Q) = \pm 1$

**Ejemplo:** Matrices de rotación son ortogonales

```python
R = rotation_matrix_2d(45)

# Verificar ortogonalidad
I = R.T @ R
print("R^T @ R:")
print(np.round(I, 10))
# [[1. 0.]
#  [0. 1.]]

print(f"det(R) = {np.linalg.det(R)}")  # 1.0
```

### 9.3 Transformaciones Rígidas

**Transformaciones rígidas** (Euclidean transforms) = Rotación + Traslación

Preservan:
- Distancias
- Ángulos
- Formas (isometrías)

---

## 10. Implementación Práctica: Sistema de Transformaciones

```python
class Transform2D:
    """Clase para manejar transformaciones 2D"""
    
    def __init__(self):
        self.matrix = np.eye(3)  # Identidad
    
    def translate(self, tx, ty):
        T = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]])
        self.matrix = self.matrix @ T
        return self
    
    def rotate(self, degrees):
        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        self.matrix = self.matrix @ R
        return self
    
    def scale(self, sx, sy=None):
        if sy is None:
            sy = sx
        S = np.array([[sx, 0, 0],
                      [0, sy, 0],
                      [0, 0, 1]])
        self.matrix = self.matrix @ S
        return self
    
    def apply(self, points):
        """
        Aplica transformación a puntos
        points: (2, N) o (N, 2) array
        """
        if points.shape[0] == 2:
            # (2, N) format
            points_h = np.vstack([points, np.ones(points.shape[1])])
        else:
            # (N, 2) format
            points_h = np.column_stack([points, np.ones(len(points))]).T
        
        transformed = self.matrix @ points_h
        
        if points.shape[0] == 2:
            return transformed[:2]
        else:
            return transformed[:2].T
    
    def reset(self):
        self.matrix = np.eye(3)
        return self

# Ejemplo de uso
transform = Transform2D()
transform.translate(2, 3).rotate(45).scale(1.5)

# Aplicar a un cuadrado
square = np.array([[0, 1, 1, 0],
                   [0, 0, 1, 1]])

transformed_square = transform.apply(square)

print("Matriz de transformación total:")
print(transform.matrix)
print("\nCuadrado transformado:")
print(transformed_square)
```

---

## 11. Ejercicios Prácticos

### Ejercicio 1: Animación de Rotación
Crea una animación que rote un triángulo 360° alrededor de su centroide.

### Ejercicio 2: Espejo
Implementa una función que refleje una forma sobre una línea arbitraria $y = mx + b$.

### Ejercicio 3: Data Augmentation
Crea un pipeline de augmentation que:
- Rote aleatoriamente ±20°
- Escale entre 0.8× y 1.2×
- Traslade hasta ±15 píxeles
- Aplique cizallamiento aleatorio

### Ejercicio 4: Transformación Inversa
Dada una transformación que rota 30° y traslada (5, -3), calcula la transformación inversa que deshace estos cambios.

```python
# Pista:
M_inv = np.linalg.inv(M)
```

### Ejercicio 5: Proyección Ortográfica
Implementa una matriz de proyección ortográfica 3D → 2D que proyecte sobre el plano XY.

---

## 12. Errores Comunes

### ❌ Error 1: Orden Incorrecto de Transformaciones
```python
# ❌ Incorrecto (traslada antes de rotar)
M = R @ T  

# ✅ Correcto (rota antes de trasladar)
M = T @ R
```

### ❌ Error 2: Olvidar Coordenadas Homogéneas para Traslación
```python
# ❌ No funciona (2×2 no puede trasladar)
T = np.array([[1, 0],
              [0, 1]])

# ✅ Correcto (3×3 con coordenadas homogéneas)
T = np.array([[1, 0, tx],
              [0, 1, ty],
              [0, 0, 1]])
```

### ❌ Error 3: No Normalizar después de Perspectiva
```python
# Después de transformación perspectiva
p_transformed = H @ p

# ❌ Usar directamente
# x, y = p_transformed[0], p_transformed[1]

# ✅ Normalizar por componente homogénea
x = p_transformed[0] / p_transformed[2]
y = p_transformed[1] / p_transformed[2]
```

---

## 13. Recursos Adicionales

### 📚 Libros
- **"Multiple View Geometry"** (Hartley & Zisserman) - Transformaciones en CV
- **"Fundamentals of Computer Graphics"** (Shirley) - Gráficos 3D

### 📺 Videos
- **3Blue1Brown:** "Linear transformations and matrices"
- **Khan Academy:** "Linear Algebra - Transformations"

### 🛠️ Bibliotecas
- **OpenCV:** `cv2.warpAffine()`, `cv2.warpPerspective()`
- **scikit-image:** `transform` module
- **PyTorch:** Spatial Transformer Networks

---

## 📌 Resumen Clave

| Transformación | Matriz 2D | Preserva | Aplicación |
|----------------|-----------|----------|------------|
| **Escalado** | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$ | Ángulos (si uniforme) | Redimensionar |
| **Rotación** | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | Distancias, ángulos | Orientación |
| **Reflexión** | $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$ | Distancias | Simetría |
| **Traslación** | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | Todo (rígida) | Posición |
| **Cizallamiento** | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ | Área | Distorsión |

**Recuerda:** 
- Coordenadas homogéneas permiten representar traslación
- Orden de transformaciones importa: $M = T_3 \cdot T_2 \cdot T_1$
- Aplicación: de derecha a izquierda

---

## 🎯 Conclusión

Has completado las **Semanas 1-2: Álgebra Lineal**! 

**Lo que dominaste:**
✅ Vectores y operaciones vectoriales
✅ Matrices y multiplicación matricial
✅ Sistemas de ecuaciones lineales
✅ NumPy para álgebra lineal
✅ Transformaciones geométricas

**Próximo módulo:**
**Semanas 3-4:** Álgebra Lineal Avanzada
- Eigenvalues y Eigenvectors
- Descomposición SVD
- PCA (Principal Component Analysis)
- Aplicaciones en reducción de dimensionalidad

---

*Las transformaciones lineales son omnipresentes en ML: desde data augmentation en visión hasta atención en transformers. ¡Dominar este concepto te da superpoderes!* 🚀
