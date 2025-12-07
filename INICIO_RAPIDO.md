# 🚀 Guía de Inicio Rápido

## ¡Bienvenido a tu Programa de IA!

Sigue estos pasos para configurar todo y empezar tu aprendizaje.

---

## 📋 Paso 1: Verifica tu Python

Abre PowerShell y ejecuta:

```powershell
python --version
```

**Necesitas Python 3.10 o superior**. Si no lo tienes:
- Descarga desde: https://www.python.org/downloads/
- Durante la instalación, marca "Add Python to PATH"

---

## 🔧 Paso 2: Crea un Entorno Virtual

En la carpeta de este proyecto, ejecuta:

```powershell
# Navega a la carpeta del proyecto
cd "c:\Users\gmfe2\OneDrive\Documentos\Código\IA"

# Crea el entorno virtual
python -m venv venv

# Activa el entorno virtual
.\venv\Scripts\Activate.ps1
```

Si tienes problemas de permisos, ejecuta primero:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Verás `(venv)` al inicio de tu línea de comandos cuando esté activado.

---

## 📦 Paso 3: Instala las Dependencias

Con el entorno virtual activado:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instalará todas las librerías necesarias (puede tomar 5-10 minutos).

---

## ✅ Paso 4: Verifica la Instalación

Ejecuta este comando para verificar que todo funciona:

```powershell
python -c "import numpy; import pandas; import torch; import sklearn; print('✅ ¡Todo instalado correctamente!')"
```

---

## 📚 Paso 5: Empieza tu Primera Lección

¡Ya estás listo! Ahora:

1. **Lee el plan semanal**: `PLAN_SEMANAL.md`
2. **Ve a la Semana 1**: `01_Fundamentos_Matematicos/Algebra_Lineal/`
3. **Lee la teoría**: `teoria/01_vectores_fundamentos.md`
4. **Haz los ejercicios**: `ejercicios/dia_01_vectores.py`

---

## 🛠️ Configuración de VS Code (Recomendado)

### Extensiones Útiles

Instala estas extensiones en VS Code:
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Jupyter** (Microsoft)
- **GitLens** (opcional pero útil)

### Selecciona el Intérprete

1. Presiona `Ctrl+Shift+P`
2. Escribe "Python: Select Interpreter"
3. Elige el que dice `venv` (./venv/Scripts/python.exe)

---

## 📝 Rutina Diaria Recomendada

```
1. Activa el entorno virtual
2. Lee la teoría del día (30-45 min)
3. Resuelve los ejercicios (1-2 horas)
4. Experimenta y haz preguntas
5. Documenta lo que aprendiste
```

---

## 🆘 Solución de Problemas

### "No se puede ejecutar scripts en este sistema"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Python no se reconoce como comando"
- Reinstala Python marcando "Add to PATH"
- O usa la ruta completa: `C:\Python3XX\python.exe`

### "pip install falla"
- Actualiza pip: `python -m pip install --upgrade pip`
- Si un paquete específico falla, instálalo por separado

### "Torch no tiene CUDA"
- La versión CPU de PyTorch es suficiente para empezar
- Más adelante puedes instalar la versión CUDA si tienes GPU NVIDIA

---

## 📊 Seguimiento de Progreso

Actualiza tu progreso en `PLAN_SEMANAL.md`:
- Marca las casillas ✅ cuando completes temas
- Anota tus horas de estudio
- Documenta dudas o dificultades

---

## 💬 Cómo Pedirme Ayuda

**Cuando tengas dudas, dime**:
1. ¿Qué estás intentando hacer?
2. ¿Qué error estás obteniendo?
3. ¿Qué has intentado ya?

**Ejemplo bueno**:
> "Estoy en el ejercicio día 1 de vectores. Mi función `magnitude` retorna 25 en lugar de 5 para el vector [3, 4]. No entiendo por qué."

---

## 🎯 ¡Estás Listo!

Tu próxima acción:
```powershell
# 1. Activa el entorno
.\venv\Scripts\Activate.ps1

# 2. Abre el primer ejercicio
code "01_Fundamentos_Matematicos\Algebra_Lineal\ejercicios\dia_01_vectores.py"
```

**¡Empecemos a aprender IA! 🚀**

---

## 📅 Recordatorios

- [ ] Estudia al menos 2 horas diarias
- [ ] Completa todos los ejercicios antes de avanzar
- [ ] Haz los proyectos semanales
- [ ] No te desanimes si algo es difícil, ¡pregúntame!

**Próxima lectura**: `01_Fundamentos_Matematicos/Algebra_Lineal/README.md`
