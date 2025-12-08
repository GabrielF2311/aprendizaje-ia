"""
ÁLGEBRA LINEAL - DÍA 7: PROYECTO - Transformaciones 2D
=======================================================

Implementa un sistema completo de transformaciones geométricas.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ============================================================================
# PARTE 1: Matrices de Transformación
# ============================================================================

def rotation_matrix(angle_degrees: float) -> np.ndarray:
    """
    Crea una matriz de rotación 2D.
    
    Args:
        angle_degrees: Ángulo en grados (positivo = antihorario)
        
    Returns:
        Matriz 2x2 de rotación
    """
    # TODO: Implementa la matriz de rotación
    # Fórmula: [[cos(θ), -sin(θ)],
    #           [sin(θ),  cos(θ)]]
    pass


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """Matriz de escalado"""
    # TODO: Implementa [[sx, 0], [0, sy]]
    pass


def reflection_matrix(axis: str) -> np.ndarray:
    """
    Matriz de reflexión.
    
    Args:
        axis: 'x' o 'y'
    """
    # TODO: Implementa reflexión
    # Reflexión en X: [[1, 0], [0, -1]]
    # Reflexión en Y: [[-1, 0], [0, 1]]
    pass


# ============================================================================
# PARTE 2: Aplicar Transformaciones
# ============================================================================

def apply_transformation(points: np.ndarray, 
                        transformation: np.ndarray) -> np.ndarray:
    """
    Aplica una transformación a un conjunto de puntos.
    
    Args:
        points: Array de forma (2, n) donde cada columna es un punto [x, y]
        transformation: Matriz 2x2
        
    Returns:
        Puntos transformados
    """
    # TODO: Multiplica transformation @ points
    pass


def compose_transformations(*matrices: np.ndarray) -> np.ndarray:
    """
    Compone múltiples transformaciones en una sola matriz.
    
    Las transformaciones se aplican de derecha a izquierda.
    """
    # TODO: Multiplica todas las matrices
    pass


# ============================================================================
# PARTE 3: Figuras Geométricas
# ============================================================================

def create_triangle() -> np.ndarray:
    """Crea un triángulo simple"""
    return np.array([
        [0, 2, 1],    # x coordinates
        [0, 0, 2]     # y coordinates
    ])


def create_square(size: float = 2) -> np.ndarray:
    """Crea un cuadrado"""
    # TODO: Implementa un cuadrado de tamaño size
    pass


def create_pentagon() -> np.ndarray:
    """Crea un pentágono regular"""
    angles = np.linspace(0, 2*np.pi, 6)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.array([x, y])


# ============================================================================
# PARTE 4: Visualización
# ============================================================================

def plot_transformation(original: np.ndarray,
                       transformed: np.ndarray,
                       title: str = "Transformación"):
    """
    Visualiza una transformación.
    """
    plt.figure(figsize=(12, 6))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.plot(original[0], original[1], 'b-o', linewidth=2, label='Original')
    plt.plot([original[0][-1], original[0][0]], 
             [original[1][-1], original[1][0]], 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Original')
    
    # Transformada
    plt.subplot(1, 2, 2)
    plt.plot(transformed[0], transformed[1], 'r-o', linewidth=2, label='Transformada')
    plt.plot([transformed[0][-1], transformed[0][0]], 
             [transformed[1][-1], transformed[1][0]], 'r-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Transformada')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_both(original: np.ndarray,
              transformed: np.ndarray,
              title: str = "Comparación"):
    """Muestra ambas figuras en el mismo gráfico"""
    plt.figure(figsize=(8, 8))
    
    # Original
    plt.plot(original[0], original[1], 'b-o', linewidth=2, label='Original', alpha=0.7)
    plt.plot([original[0][-1], original[0][0]], 
             [original[1][-1], original[1][0]], 'b-', linewidth=2, alpha=0.7)
    
    # Transformada
    plt.plot(transformed[0], transformed[1], 'r-o', linewidth=2, label='Transformada', alpha=0.7)
    plt.plot([transformed[0][-1], transformed[0][0]], 
             [transformed[1][-1], transformed[1][0]], 'r-', linewidth=2, alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.legend()
    plt.title(title)
    plt.show()


# ============================================================================
# PARTE 5: Demos
# ============================================================================

def demo_rotation():
    """Demo: Rotación de un triángulo"""
    print("🔄 Demo: Rotación 45°")
    
    # TODO: Crea triángulo
    triangle = create_triangle()
    
    # TODO: Crea matriz de rotación
    R = rotation_matrix(45)
    
    # TODO: Aplica transformación
    rotated = apply_transformation(triangle, R)
    
    # TODO: Visualiza
    plot_both(triangle, rotated, "Rotación 45°")


def demo_scaling():
    """Demo: Escalado"""
    # TODO: Implementa demo de escalado
    pass


def demo_reflection():
    """Demo: Reflexión"""
    # TODO: Implementa demo de reflexión
    pass


def demo_composition():
    """
    Demo: Composición de transformaciones
    
    Aplica: Rotar 30° → Escalar 1.5x → Reflejar en Y
    """
    # TODO: Implementa composición
    pass


# ============================================================================
# DESAFÍOS OPCIONALES
# ============================================================================

def challenge_animation():
    """
    DESAFÍO: Crea una animación de rotación continua.
    """
    from matplotlib.animation import FuncAnimation
    
    # TODO: Implementa animación
    # Pista: Rota el triángulo de 0° a 360° en pasos pequeños
    pass


def challenge_custom_shape():
    """
    DESAFÍO: Crea tu propia figura personalizada.
    """
    # TODO: Crea una letra, símbolo, o figura creativa
    pass


# ============================================================================
# MAIN - PROYECTO COMPLETO
# ============================================================================

def run_project():
    """Ejecuta todas las demos del proyecto"""
    print("=" * 60)
    print("PROYECTO: TRANSFORMACIONES GEOMÉTRICAS 2D")
    print("=" * 60)
    print()
    
    # Descomenta para ejecutar:
    # demo_rotation()
    # demo_scaling()
    # demo_reflection()
    # demo_composition()


if __name__ == "__main__":
    run_project()
    
    print("\n✅ Proyecto completado!")
    print("💡 Intenta los desafíos opcionales si quieres más práctica")
