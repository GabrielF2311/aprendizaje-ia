# ⚙️ Configuración del Entorno de Desarrollo

Esta guía te ayudará a configurar todo lo necesario para empezar.

## 📋 Checklist Pre-requisitos

- [ ] Windows 10/11
- [ ] Python 3.10 o superior instalado
- [ ] VS Code instalado
- [ ] Git instalado
- [ ] Al menos 10 GB de espacio libre en disco

---

## 🐍 Paso 1: Instalar Python

### Verificar si ya tienes Python

```powershell
python --version
```

Si muestra Python 3.10 o superior, ¡perfecto! Si no:

### Instalar Python

1. Ve a: https://www.python.org/downloads/
2. Descarga Python 3.11 o superior
3. **IMPORTANTE**: Durante la instalación:
   - ✅ Marca "Add Python to PATH"
   - ✅ Marca "Install for all users" (opcional)
4. Verifica: `python --version`

---

## 🔧 Paso 2: Configurar Entorno Virtual

### ¿Por qué un entorno virtual?

- Aísla las dependencias de este proyecto
- Evita conflictos con otros proyectos
- Fácil de replicar en otras máquinas

### Crear el entorno

```powershell
# Navega a la carpeta del proyecto
cd "c:\Users\gmfe2\OneDrive\Documentos\Código\IA"

# Crea el entorno virtual
python -m venv venv
```

Esto crea una carpeta `venv/` con Python y pip aislados.

### Activar el entorno

```powershell
# En PowerShell
.\venv\Scripts\Activate.ps1
```

Si obtienes error de permisos:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego intenta activar de nuevo.

### Verificar activación

Deberías ver `(venv)` al inicio de tu prompt:
```
(venv) PS C:\Users\gmfe2\OneDrive\Documentos\Código\IA>
```

### Desactivar (cuando termines)

```powershell
deactivate
```

---

## 📦 Paso 3: Instalar Dependencias

Con el entorno virtual ACTIVADO:

```powershell
# Actualiza pip
python -m pip install --upgrade pip

# Instala todas las dependencias
pip install -r requirements.txt
```

**Tiempo estimado**: 5-10 minutos (depende de tu internet)

### Verificar instalación

```powershell
python -c "import numpy, pandas, torch, sklearn; print('✅ Todo instalado correctamente!')"
```

Si ves el mensaje de éxito, ¡listo!

---

## 💻 Paso 4: Configurar VS Code

### Instalar VS Code

Si no lo tienes: https://code.visualstudio.com/

### Extensiones Esenciales

Abre VS Code y presiona `Ctrl+Shift+X` para abrir extensiones:

1. **Python** (Microsoft) - ID: `ms-python.python`
   - Soporte completo para Python
   - IntelliSense, debugging, linting

2. **Pylance** (Microsoft) - ID: `ms-python.vscode-pylance`
   - Type checking avanzado
   - Mejor autocompletado

3. **Jupyter** (Microsoft) - ID: `ms-toolsai.jupyter`
   - Notebooks en VS Code
   - Visualización inline

### Extensiones Recomendadas

4. **GitLens** - ID: `eamodio.gitlens`
   - Mejora la experiencia con Git

5. **Python Indent** - ID: `KevinRose.vsc-python-indent`
   - Auto-indentación inteligente

6. **autoDocstring** - ID: `njpwerner.autodocstring`
   - Genera docstrings automáticamente

### Seleccionar el Intérprete de Python

1. Presiona `Ctrl+Shift+P`
2. Escribe: "Python: Select Interpreter"
3. Selecciona el que dice `venv` o muestra la ruta `.\venv\Scripts\python.exe`

---

## 🎨 Paso 5: Configurar Settings de VS Code

Crea/edita `.vscode/settings.json` en tu workspace:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    },
    "jupyter.askForKernelRestart": false
}
```

---

## 🔥 Paso 6: Configurar PyTorch

### CPU vs GPU

Por defecto, `requirements.txt` instala PyTorch con soporte CPU.

### Si tienes GPU NVIDIA

Verifica CUDA:
```powershell
nvidia-smi
```

Si tienes CUDA 11.8 o 12.x:
```powershell
# Desinstala la versión CPU
pip uninstall torch torchvision torchaudio

# Instala versión GPU (CUDA 12.1 ejemplo)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Para otras versiones de CUDA: https://pytorch.org/get-started/locally/

### Verificar PyTorch

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

---

## 📊 Paso 7: Verificar Todo

Ejecuta este script de verificación:

```python
# verifica_instalacion.py
import sys

def check_package(name, import_name=None):
    """Verifica si un paquete está instalado"""
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'desconocida')
        print(f"✅ {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: NO INSTALADO")
        return False

print("=" * 60)
print("VERIFICACIÓN DE ENTORNO")
print("=" * 60)
print()

print(f"Python: {sys.version}")
print()

packages = [
    ('NumPy', 'numpy'),
    ('Pandas', 'pandas'),
    ('Matplotlib', 'matplotlib'),
    ('Seaborn', 'seaborn'),
    ('Scikit-learn', 'sklearn'),
    ('PyTorch', 'torch'),
    ('TorchVision', 'torchvision'),
    ('Transformers', 'transformers'),
    ('Jupyter', 'jupyter'),
]

print("Paquetes instalados:")
print("-" * 60)

all_ok = all(check_package(name, imp) for name, imp in packages)

print()
if all_ok:
    print("🎉 ¡Todo está correctamente instalado!")
    print("✅ Listo para empezar a aprender IA")
else:
    print("⚠️ Algunos paquetes faltan. Ejecuta:")
    print("   pip install -r requirements.txt")
```

Guarda como `verifica_instalacion.py` y ejecuta:
```powershell
python verifica_instalacion.py
```

---

## 🗂️ Paso 8: Organización del Workspace

Tu estructura debe verse así:

```
IA/
├── venv/                          # Entorno virtual (no subir a git)
├── .vscode/                       # Configuración de VS Code
│   └── settings.json
├── 01_Fundamentos_Matematicos/
├── 02_Python_para_IA/
├── 03_Machine_Learning/
├── 04_Deep_Learning/
├── 05_Proyectos/
├── 06_Recursos/
├── 07_Datasets/                   # Datasets (no subir a git)
├── .gitignore
├── requirements.txt
├── README.md
├── PLAN_SEMANAL.md
├── INICIO_RAPIDO.md
└── MI_PROGRESO.md
```

---

## 🌐 Paso 9: Configurar Git (Opcional pero Recomendado)

### Instalar Git

https://git-scm.com/downloads

### Configurar Git

```powershell
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```

### Inicializar repositorio

```powershell
cd "c:\Users\gmfe2\OneDrive\Documentos\Código\IA"
git init
git add .
git commit -m "Initial commit: Estructura del programa de IA"
```

### Crear repositorio en GitHub (opcional)

1. Ve a github.com
2. Crea un nuevo repositorio
3. Conecta tu repo local:

```powershell
git remote add origin https://github.com/GabrielF2311/aprendizaje-ia.git
git branch -M main
git push -u origin main
```

---

## 🧪 Paso 10: Test Drive

Prueba tu configuración con este notebook:

```python
# test_environment.ipynb

# Celda 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print("✅ Imports exitosos")

# Celda 2: NumPy
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {arr}")
print(f"Promedio: {arr.mean()}")

# Celda 3: Pandas
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print(df)

# Celda 4: Matplotlib
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title('Test Plot')
plt.show()

# Celda 5: PyTorch
x = torch.tensor([1.0, 2.0, 3.0])
print(f"PyTorch tensor: {x}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

print("\n🎉 ¡Todo funciona correctamente!")
```

---

## 🆘 Solución de Problemas Comunes

### "python no se reconoce como comando"

**Solución**:
1. Reinstala Python marcando "Add to PATH"
2. O agrega manualmente:
   - Busca "variables de entorno"
   - Edita PATH
   - Agrega: `C:\Users\TU_USUARIO\AppData\Local\Programs\Python\Python311`

### "No se pueden ejecutar scripts en este sistema"

**Solución**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error al instalar PyTorch

**Solución**:
```powershell
# Instala solo la versión CPU primero
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### VS Code no encuentra el intérprete

**Solución**:
1. `Ctrl+Shift+P`
2. "Python: Select Interpreter"
3. Si no aparece, selecciona "Enter interpreter path"
4. Navega a `venv\Scripts\python.exe`

### Jupyter no funciona en VS Code

**Solución**:
```powershell
pip install ipykernel
python -m ipykernel install --user --name=venv
```

---

## ✅ Checklist Final

- [ ] Python 3.10+ instalado
- [ ] Entorno virtual creado y activado
- [ ] Todas las dependencias instaladas
- [ ] VS Code configurado con extensiones
- [ ] Intérprete de Python seleccionado en VS Code
- [ ] Script de verificación ejecutado exitosamente
- [ ] Git configurado (opcional)
- [ ] Test notebook funciona correctamente

---

## 🎯 Próximos Pasos

1. ✅ Lee `INICIO_RAPIDO.md`
2. ✅ Revisa `PLAN_SEMANAL.md`
3. ✅ Empieza con Semana 1: Álgebra Lineal
4. ✅ Mantén actualizado `MI_PROGRESO.md`

---

**¡Felicidades! Tu entorno está listo. Ahora a aprender IA! 🚀**
