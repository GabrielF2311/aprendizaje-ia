"""
ÁLGEBRA LINEAL - DÍA 1: VECTORES
================================

Ejercicios para entender vectores desde cero.
NO uses NumPy en esta parte, implementa todo desde cero.

Objetivos:
- Crear vectores
- Calcular norma/magnitud
- Normalizar vectores
- Entender representación geométrica
"""

import math
from typing import List, Union

# ============================================================================
# EJERCICIO 1: Clase Vector
# ============================================================================

class Vector:
    """
    Implementa una clase Vector que soporte operaciones básicas.
    
    Ejemplo de uso:
        v = Vector([1, 2, 3])
        print(v)  # Vector([1, 2, 3])
    """
    
    def __init__(self, components: List[float]):
        """
        Inicializa un vector con sus componentes.
        
        Args:
            components: Lista de números representando el vector
        """
        # TODO: Implementa esto
        pass
    
    def __str__(self):
        """Representación en string del vector"""
        # TODO: Implementa esto
        pass
    
    def __repr__(self):
        """Representación para debugging"""
        return self.__str__()
    
    def dimension(self) -> int:
        """Retorna la dimensión del vector"""
        # TODO: Implementa esto
        pass


# ============================================================================
# EJERCICIO 2: Magnitud/Norma del Vector
# ============================================================================

def magnitude(vector: List[float]) -> float:
    """
    Calcula la magnitud (norma L2) de un vector.
    
    Fórmula: ||v|| = sqrt(v1² + v2² + ... + vn²)
    
    Args:
        vector: Lista de componentes del vector
        
    Returns:
        La magnitud del vector
        
    Ejemplo:
        >>> magnitude([3, 4])
        5.0
        >>> magnitude([1, 2, 2])
        3.0
    """
    # TODO: Implementa esto
    # Pista: usa sum() y math.sqrt()
    pass


# ============================================================================
# EJERCICIO 3: Normalizar Vector
# ============================================================================

def normalize(vector: List[float]) -> List[float]:
    """
    Normaliza un vector (lo convierte en vector unitario).
    
    Un vector normalizado tiene magnitud 1.
    Fórmula: v_norm = v / ||v||
    
    Args:
        vector: Vector a normalizar
        
    Returns:
        Vector normalizado
        
    Ejemplo:
        >>> normalize([3, 4])
        [0.6, 0.8]
    """
    # TODO: Implementa esto
    # Pista: divide cada componente por la magnitud
    pass


# ============================================================================
# EJERCICIO 4: Verificar si es Vector Unitario
# ============================================================================

def is_unit_vector(vector: List[float], tolerance: float = 1e-10) -> bool:
    """
    Verifica si un vector es unitario (magnitud = 1).
    
    Args:
        vector: Vector a verificar
        tolerance: Tolerancia para comparación de flotantes
        
    Returns:
        True si es vector unitario, False en caso contrario
        
    Ejemplo:
        >>> is_unit_vector([1, 0, 0])
        True
        >>> is_unit_vector([0.6, 0.8])
        True
        >>> is_unit_vector([1, 1])
        False
    """
    # TODO: Implementa esto
    pass


# ============================================================================
# EJERCICIO 5: Distancia entre Vectores
# ============================================================================

def distance(v1: List[float], v2: List[float]) -> float:
    """
    Calcula la distancia euclidiana entre dos vectores.
    
    Fórmula: d(v1, v2) = ||v1 - v2||
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        Distancia entre los vectores
        
    Ejemplo:
        >>> distance([1, 2], [4, 6])
        5.0
    """
    # TODO: Implementa esto
    # Pista: calcula la magnitud de la diferencia
    pass


# ============================================================================
# EJERCICIO 6: Crear Vectores Especiales
# ============================================================================

def zero_vector(dimension: int) -> List[float]:
    """
    Crea un vector de ceros de dimensión dada.
    
    Ejemplo:
        >>> zero_vector(3)
        [0.0, 0.0, 0.0]
    """
    # TODO: Implementa esto
    pass


def ones_vector(dimension: int) -> List[float]:
    """
    Crea un vector de unos de dimensión dada.
    
    Ejemplo:
        >>> ones_vector(4)
        [1.0, 1.0, 1.0, 1.0]
    """
    # TODO: Implementa esto
    pass


def standard_basis_vector(dimension: int, index: int) -> List[float]:
    """
    Crea un vector de la base estándar.
    (Un vector con 1 en la posición 'index' y 0 en el resto)
    
    Args:
        dimension: Dimensión del vector
        index: Índice donde colocar el 1 (0-indexed)
        
    Ejemplo:
        >>> standard_basis_vector(3, 1)
        [0.0, 1.0, 0.0]
    """
    # TODO: Implementa esto
    pass


# ============================================================================
# TESTS - Verifica tus implementaciones
# ============================================================================

def run_tests():
    """Ejecuta tests básicos de tus funciones"""
    
    print("🧪 Ejecutando tests...\n")
    
    # Test 1: Magnitud
    print("Test 1: Magnitud")
    assert abs(magnitude([3, 4]) - 5.0) < 1e-10, "Error en magnitude con [3, 4]"
    assert abs(magnitude([1, 2, 2]) - 3.0) < 1e-10, "Error en magnitude con [1, 2, 2]"
    print("✅ Magnitud funciona correctamente\n")
    
    # Test 2: Normalización
    print("Test 2: Normalización")
    norm = normalize([3, 4])
    assert abs(magnitude(norm) - 1.0) < 1e-10, "Vector normalizado no tiene magnitud 1"
    print("✅ Normalización funciona correctamente\n")
    
    # Test 3: Vector unitario
    print("Test 3: Vector Unitario")
    assert is_unit_vector([1, 0, 0]) == True, "Error: [1,0,0] debería ser unitario"
    assert is_unit_vector([1, 1]) == False, "Error: [1,1] no debería ser unitario"
    print("✅ Verificación de vector unitario funciona\n")
    
    # Test 4: Distancia
    print("Test 4: Distancia")
    assert abs(distance([1, 2], [4, 6]) - 5.0) < 1e-10, "Error en cálculo de distancia"
    print("✅ Distancia funciona correctamente\n")
    
    # Test 5: Vectores especiales
    print("Test 5: Vectores especiales")
    assert zero_vector(3) == [0.0, 0.0, 0.0], "Error en zero_vector"
    assert ones_vector(2) == [1.0, 1.0], "Error en ones_vector"
    assert standard_basis_vector(3, 1) == [0.0, 1.0, 0.0], "Error en standard_basis_vector"
    print("✅ Vectores especiales funcionan correctamente\n")
    
    print("🎉 ¡Todos los tests pasaron! ¡Excelente trabajo!")


# ============================================================================
# DESAFÍO ADICIONAL (OPCIONAL)
# ============================================================================

def challenge_angle_between_vectors(v1: List[float], v2: List[float]) -> float:
    """
    DESAFÍO: Calcula el ángulo entre dos vectores en radianes.
    
    Pista: Necesitarás el producto punto (lo veremos mañana)
    Fórmula: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
    
    No te preocupes si no puedes hacerlo aún, lo veremos mañana.
    """
    # OPCIONAL - Intenta si quieres un desafío extra
    pass


# ============================================================================
# MAIN - Ejecuta este archivo para probar tu código
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ÁLGEBRA LINEAL - DÍA 1: VECTORES")
    print("=" * 60)
    print()
    
    # Primero, implementa todas las funciones arriba
    # Luego, descomenta la siguiente línea para ejecutar los tests
    # run_tests()
    
    # Experimenta con vectores aquí
    print("🔬 Experimenta con tus funciones:\n")
    
    # Ejemplo de uso (descomenta cuando implementes las funciones):
    # v1 = [3, 4]
    # print(f"Vector: {v1}")
    # print(f"Magnitud: {magnitude(v1)}")
    # print(f"Normalizado: {normalize(v1)}")
    # print(f"Es unitario: {is_unit_vector(v1)}")
