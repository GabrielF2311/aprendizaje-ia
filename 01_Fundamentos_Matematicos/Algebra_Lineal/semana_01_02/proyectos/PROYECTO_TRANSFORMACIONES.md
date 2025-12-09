# 🎯 Proyecto: Sistema de Transformaciones 2D

## Descripción

Implementarás un sistema que aplica transformaciones geométricas a figuras 2D usando matrices. Este proyecto consolida todo lo aprendido en las primeras dos semanas.

## 🎓 Conceptos que Practicarás

- Multiplicación de matrices
- Transformaciones lineales
- Visualización con matplotlib
- Composición de transformaciones
- NumPy para álgebra lineal

## 📋 Requisitos

### Parte 1: Implementar Transformaciones (60%)

Crea funciones que retornen matrices de transformación:

1. **Rotación**: Rotar puntos alrededor del origen
2. **Escalado**: Cambiar el tamaño de figuras
3. **Reflexión**: Reflejar respecto a un eje
4. **Traslación**: Mover figuras (usando coordenadas homogéneas)

### Parte 2: Aplicar Transformaciones (20%)

- Función que aplica una transformación a un conjunto de puntos
- Función que compone múltiples transformaciones

### Parte 3: Visualización (20%)

- Graficar la figura original
- Graficar la figura transformada
- Mostrar ambas en el mismo gráfico

## 🔨 Especificaciones Técnicas

### Matrices de Transformación 2D

**Rotación** (θ radianes, sentido antihorario):
```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

**Escalado** (sx, sy factores de escala):
```
S(sx, sy) = [sx   0]
            [0   sy]
```

**Reflexión en eje X**:
```
Fx = [1   0]
     [0  -1]
```

**Reflexión en eje Y**:
```
Fy = [-1  0]
     [0   1]
```

**Traslación** (usando coordenadas homogéneas):
```
T(tx, ty) = [1  0  tx]
            [0  1  ty]
            [0  0   1]
```

## 💻 Template de Código

```python
"""
PROYECTO SEMANA 1-2: TRANSFORMACIONES 2D
=========================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ============================================================================
# PARTE 1: MATRICES DE TRANSFORMACIÓN
# ============================================================================

def rotation_matrix(angle_degrees: float) -> np.ndarray:
    """
    Crea matriz de rotación 2D.
    
    Args:
        angle_degrees: Ángulo en grados (positivo = antihorario)
        
    Returns:
        Matriz 2x2 de rotación
        
    Ejemplo:
        >>> R = rotation_matrix(90)
        >>> # Rota 90° antihorario
    """
    # TODO: Implementa esto
    # Pistas:
    # 1. Convierte grados a radianes: np.radians()
    # 2. Usa np.cos() y np.sin()
    # 3. Retorna matriz 2x2
    pass


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """
    Crea matriz de escalado.
    
    Args:
        sx: Factor de escala en X
        sy: Factor de escala en Y
        
    Returns:
        Matriz 2x2 de escalado
    """
    # TODO: Implementa esto
    pass


def reflection_matrix(axis: str) -> np.ndarray:
    """
    Crea matriz de reflexión.
    
    Args:
        axis: 'x' o 'y' indicando eje de reflexión
        
    Returns:
        Matriz 2x2 de reflexión
    """
    # TODO: Implementa esto
    pass


# ============================================================================
# PARTE 2: APLICAR TRANSFORMACIONES
# ============================================================================

def apply_transformation(points: np.ndarray, 
                        transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Aplica una transformación a un conjunto de puntos.
    
    Args:
        points: Array de forma (2, n) donde cada columna es un punto [x, y]
        transformation_matrix: Matriz 2x2 de transformación
        
    Returns:
        Puntos transformados (2, n)
        
    Ejemplo:
        >>> triangle = np.array([[0, 1, 0.5], [0, 0, 1]])
        >>> R = rotation_matrix(45)
        >>> rotated = apply_transformation(triangle, R)
    """
    # TODO: Implementa esto
    # Pista: Usa multiplicación de matrices (@ o np.dot)
    pass


def compose_transformations(*matrices: np.ndarray) -> np.ndarray:
    """
    Compone múltiples transformaciones en una sola matriz.
    
    Args:
        *matrices: Matrices de transformación a componer
        
    Returns:
        Matriz resultante de la composición
        
    Nota: Las transformaciones se aplican de derecha a izquierda
    """
    # TODO: Implementa esto
    # Pista: Multiplica todas las matrices en orden
    pass


# ============================================================================
# PARTE 3: VISUALIZACIÓN
# ============================================================================

def plot_transformation(original: np.ndarray, 
                       transformed: np.ndarray,
                       title: str = "Transformación"):
    """
    Visualiza figura original y transformada.
    
    Args:
        original: Puntos originales (2, n)
        transformed: Puntos transformados (2, n)
        title: Título del gráfico
    """
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Original
    plt.subplot(1, 2, 1)
    plt.plot(original[0], original[1], 'b-o', label='Original')
    plt.plot([original[0][-1], original[0][0]], 
             [original[1][-1], original[1][0]], 'b-')  # Cierra la figura
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Original')
    
    # Subplot 2: Transformada
    plt.subplot(1, 2, 2)
    plt.plot(transformed[0], transformed[1], 'r-o', label='Transformada')
    plt.plot([transformed[0][-1], transformed[0][0]], 
             [transformed[1][-1], transformed[1][0]], 'r-')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Transformada')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_comparison(original: np.ndarray, 
                   transformed: np.ndarray,
                   title: str = "Comparación"):
    """
    Visualiza ambas figuras en el mismo gráfico.
    
    Args:
        original: Puntos originales (2, n)
        transformed: Puntos transformados (2, n)
        title: Título del gráfico
    """
    plt.figure(figsize=(8, 8))
    
    # Original
    plt.plot(original[0], original[1], 'b-o', label='Original', linewidth=2)
    plt.plot([original[0][-1], original[0][0]], 
             [original[1][-1], original[1][0]], 'b-', linewidth=2)
    
    # Transformada
    plt.plot(transformed[0], transformed[1], 'r-o', label='Transformada', linewidth=2)
    plt.plot([transformed[0][-1], transformed[0][0]], 
             [transformed[1][-1], transformed[1][0]], 'r-', linewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title(title)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()


# ============================================================================
# FIGURAS DE EJEMPLO
# ============================================================================

def create_triangle() -> np.ndarray:
    """Crea un triángulo simple"""
    return np.array([
        [0, 2, 1],    # x coordinates
        [0, 0, 2]     # y coordinates
    ])


def create_square() -> np.ndarray:
    """Crea un cuadrado"""
    return np.array([
        [0, 2, 2, 0],
        [0, 0, 2, 2]
    ])


def create_house() -> np.ndarray:
    """Crea una casita"""
    return np.array([
        [0, 3, 3, 2, 1, 0, 0],  # x
        [0, 0, 2, 3, 2, 2, 0]   # y
    ])


# ============================================================================
# DEMOS Y TESTS
# ============================================================================

def demo_rotation():
    """Demo: Rotación de triángulo"""
    print("🔄 Demo: Rotación 45°")
    
    triangle = create_triangle()
    R = rotation_matrix(45)
    rotated = apply_transformation(triangle, R)
    
    plot_comparison(triangle, rotated, "Rotación 45°")


def demo_scaling():
    """Demo: Escalado"""
    print("📏 Demo: Escalado (2x, 0.5x)")
    
    square = create_square()
    S = scaling_matrix(2, 0.5)
    scaled = apply_transformation(square, S)
    
    plot_comparison(square, scaled, "Escalado (2x, 0.5x)")


def demo_reflection():
    """Demo: Reflexión"""
    print("🪞 Demo: Reflexión en eje Y")
    
    house = create_house()
    Fy = reflection_matrix('y')
    reflected = apply_transformation(house, Fy)
    
    plot_comparison(house, reflected, "Reflexión en eje Y")


def demo_composition():
    """Demo: Composición de transformaciones"""
    print("🔗 Demo: Rotar + Escalar + Reflejar")
    
    triangle = create_triangle()
    
    # Componer: Rotar 30°, luego escalar 1.5x, luego reflejar en X
    R = rotation_matrix(30)
    S = scaling_matrix(1.5, 1.5)
    Fx = reflection_matrix('x')
    
    # Composición (se aplica R, luego S, luego Fx)
    M = compose_transformations(Fx, S, R)
    
    # Aplicar
    transformed = apply_transformation(triangle, M)
    
    plot_comparison(triangle, transformed, 
                   "Rotación 30° + Escala 1.5x + Reflexión X")


# ============================================================================
# DESAFÍOS OPCIONALES
# ============================================================================

def challenge_animation():
    """
    DESAFÍO: Crea una animación de rotación continua.
    
    Pistas:
    - Usa matplotlib.animation
    - Rota en incrementos pequeños
    - Crea múltiples frames
    """
    # OPCIONAL - Intenta si quieres un desafío extra
    pass


def challenge_3d():
    """
    DESAFÍO: Extiende a transformaciones 3D.
    
    Pistas:
    - Matrices 3x3 para transformaciones
    - Usa mpl_toolkits.mplot3d para visualización
    """
    # OPCIONAL - Para los más aventureros
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROYECTO: TRANSFORMACIONES GEOMÉTRICAS 2D")
    print("=" * 60)
    print()
    
    # Descomenta las demos cuando implementes las funciones:
    
    # demo_rotation()
    # demo_scaling()
    # demo_reflection()
    # demo_composition()
    
    print("\n✅ Proyecto completado!")
    print("📝 No olvides documentar tu código y hacer un README")
```

## ✅ Criterios de Evaluación

**Funcionalidad (60%)**
- [ ] Todas las matrices de transformación funcionan correctamente
- [ ] `apply_transformation` funciona con cualquier conjunto de puntos
- [ ] `compose_transformations` combina transformaciones correctamente

**Visualización (20%)**
- [ ] Gráficos claros y bien etiquetados
- [ ] Colores diferentes para original vs transformado
- [ ] Ejes proporcionales (aspect ratio)

**Código (20%)**
- [ ] Código limpio y bien comentado
- [ ] Nombres de variables descriptivos
- [ ] Funciones con docstrings
- [ ] Uso apropiado de NumPy

## 🎁 Extras Opcionales

Si terminas rápido, intenta:

1. **Animación**: Rota una figura continuamente
2. **Interactivo**: Deslizadores para controlar transformaciones
3. **3D**: Extiende a transformaciones 3D
4. **Texto**: Transforma texto/letras

## 📤 Entrega

Cuando termines:
1. Guarda tu código en `proyecto_semana_1_2.py`
2. Crea un `README.md` explicando tu implementación
3. Incluye al menos 3 imágenes de tus visualizaciones
4. ¡Comparte tu proyecto!

## 💡 Hints

**Si te atascas**:
- Revisa la teoría de multiplicación de matrices
- Prueba cada transformación individualmente primero
- Usa figuras simples (triángulo) antes de complejas
- Verifica dimensiones de matrices con `print(matriz.shape)`

**Para debugging**:
```python
# Imprime la matriz de transformación
print("Matriz de rotación:")
print(rotation_matrix(45))

# Verifica que la figura no se deforme
original_area = ...  # Calcula el área
transformed_area = ...  # Calcula el área
print(f"Áreas: {original_area} vs {transformed_area}")
```

## 🎯 Objetivo Final

Al completar este proyecto, habrás:
- ✅ Aplicado álgebra lineal a un problema real
- ✅ Usado NumPy para computación numérica
- ✅ Visualizado resultados con matplotlib
- ✅ Entendido cómo las matrices transforman el espacio

**¡Esto es fundamental para Computer Vision y Graphics en IA!**

---

**¿Listo? ¡Empieza a programar! 🚀**
