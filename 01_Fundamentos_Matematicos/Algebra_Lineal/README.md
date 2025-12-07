# 📐 Álgebra Lineal - Semanas 1 y 2

## 🎯 Objetivos de Aprendizaje
Al finalizar estas dos semanas, serás capaz de:
- Entender y manipular vectores y matrices
- Realizar operaciones fundamentales de álgebra lineal
- Implementar estas operaciones en Python
- Comprender por qué el álgebra lineal es fundamental para IA

## 📅 Cronograma Detallado

### **Día 1: Introducción a Vectores**

**Teoría (1.5 horas)**
- ¿Qué es un vector?
- Representación geométrica vs algebraica
- Vectores en 2D y 3D
- Norma (magnitud) de un vector

**Práctica (1.5 horas)**
- Lee: `01_teoria_vectores.md`
- Resuelve: `ejercicios/dia_01_vectores.py`
- Implementa funciones de vectores sin usar librerías

**Recursos**
- Video recomendado: 3Blue1Brown - "Vectors, what even are they?"
- Lectura: Capítulo 1 del libro en `recursos/algebra_lineal_libro.pdf`

---

### **Día 2: Operaciones con Vectores**

**Teoría (1.5 horas)**
- Suma y resta de vectores
- Multiplicación por escalar
- Producto punto (dot product)
- Ángulo entre vectores
- Proyecciones

**Práctica (1.5 horas)**
- Lee: `02_operaciones_vectores.md`
- Resuelve: `ejercicios/dia_02_operaciones.py`
- Visualiza vectores con matplotlib

**Mini-desafío**
Implementa una función que calcule el ángulo entre dos vectores cualquiera.

---

### **Día 3: Introducción a Matrices**

**Teoría (1.5 horas)**
- ¿Qué es una matriz?
- Dimensiones y elementos
- Tipos de matrices (cuadrada, identidad, diagonal, triangular)
- Matrices especiales (cero, identidad)

**Práctica (1.5 horas)**
- Lee: `03_teoria_matrices.md`
- Resuelve: `ejercicios/dia_03_matrices.py`
- Crea una clase `Matrix` en Python

**Mini-desafío**
Crea una función que genere matrices de identidad de cualquier tamaño.

---

### **Día 4: Operaciones con Matrices**

**Teoría (1.5 horas)**
- Suma y resta de matrices
- Multiplicación por escalar
- Multiplicación de matrices
- Transposición
- Propiedades de las operaciones

**Práctica (1.5 horas)**
- Lee: `04_operaciones_matrices.md`
- Resuelve: `ejercicios/dia_04_operaciones_matrices.py`
- Implementa multiplicación de matrices (algoritmo O(n³))

**Mini-desafío**
Verifica las propiedades asociativa y distributiva de las matrices.

---

### **Día 5: Sistemas de Ecuaciones Lineales**

**Teoría (1.5 horas)**
- Representación matricial de sistemas lineales
- Eliminación Gaussiana
- Método de Gauss-Jordan
- Soluciones únicas, infinitas, sin solución

**Práctica (1.5 horas)**
- Lee: `05_sistemas_ecuaciones.md`
- Resuelve: `ejercicios/dia_05_sistemas.py`
- Implementa eliminación Gaussiana

**Mini-desafío**
Resuelve sistemas 3x3 con tu implementación.

---

### **Día 6: Introducción a NumPy**

**Teoría (1 hora)**
- ¿Por qué NumPy?
- Arrays vs listas de Python
- Operaciones vectorizadas
- Broadcasting

**Práctica (2 horas)**
- Lee: `06_numpy_basico.md`
- Resuelve: `ejercicios/dia_06_numpy.py`
- Reimplementa ejercicios anteriores con NumPy
- Compara velocidad: tu implementación vs NumPy

**Mini-desafío**
Mide el tiempo de multiplicación de matrices grandes (1000x1000) con tu código vs NumPy.

---

### **Día 7: PROYECTO - Transformaciones Geométricas**

**Objetivo**
Implementar un sistema de transformaciones 2D usando matrices.

**Tareas**
1. Crear funciones para:
   - Rotación
   - Escalado
   - Traslación
   - Reflexión
2. Visualizar las transformaciones con matplotlib
3. Combinar múltiples transformaciones
4. Aplicar a una figura (triángulo, cuadrado)

**Entregable**
- `proyecto_semana_1_2.py` funcionando
- Gráficos mostrando las transformaciones
- README explicando tu implementación

**Ejemplo de visualización**
```python
# Tu código debe generar algo como:
# - Figura original
# - Figura rotada 45°
# - Figura escalada 2x
# - Figura reflejada
```

---

## 📚 Recursos de Estudio

### Videos
- [3Blue1Brown - Essence of Linear Algebra (Playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Khan Academy - Linear Algebra

### Libros
- "Linear Algebra and Its Applications" - David Lay
- "Introduction to Linear Algebra" - Gilbert Strang

### Interactivos
- https://www.mathsisfun.com/algebra/matrix-introduction.html

---

## ✅ Checklist de Progreso

### Conceptos Teóricos
- [ ] Entiendo qué es un vector y cómo se representa
- [ ] Puedo calcular producto punto y norma
- [ ] Entiendo la multiplicación de matrices
- [ ] Sé resolver sistemas de ecuaciones con matrices
- [ ] Conozco matrices especiales (identidad, diagonal)

### Habilidades Prácticas
- [ ] Implementé operaciones vectoriales en Python puro
- [ ] Implementé multiplicación de matrices
- [ ] Usé NumPy para álgebra lineal
- [ ] Visualicé vectores con matplotlib
- [ ] Completé el proyecto de transformaciones

### Ejercicios Completados
- [ ] Día 1: Vectores (ejercicios/dia_01_vectores.py)
- [ ] Día 2: Operaciones vectoriales
- [ ] Día 3: Matrices básicas
- [ ] Día 4: Operaciones matriciales
- [ ] Día 5: Sistemas de ecuaciones
- [ ] Día 6: NumPy
- [ ] Día 7: Proyecto de transformaciones

---

## 🎓 Autoevaluación

Responde estas preguntas sin ver tus notas:

1. ¿Qué es el producto punto y qué información te da?
2. ¿Cuándo NO se pueden multiplicar dos matrices?
3. ¿Qué significa que una matriz sea singular?
4. ¿Para qué sirve la transposición en ML?
5. ¿Por qué NumPy es más rápido que Python puro?

**Si puedes responder 4/5, estás listo para avanzar!**

---

## 💡 Conexión con IA

**¿Por qué esto es importante para IA?**
- **Vectores**: Representan features/características de datos
- **Matrices**: Almacenan datasets completos
- **Multiplicación**: Operación fundamental en redes neuronales
- **Transposición**: Crucial en backpropagation
- **Sistemas lineales**: Base de regresión lineal

En las próximas semanas verás estos conceptos en acción!

---

## 🆘 ¿Necesitas Ayuda?

Si te atascas:
1. Revisa los ejemplos en la carpeta `ejemplos/`
2. Consulta las soluciones en `soluciones/` (¡solo después de intentarlo!)
3. Pregúntame directamente

**¡Éxito en tu primera y segunda semana! 🚀**
