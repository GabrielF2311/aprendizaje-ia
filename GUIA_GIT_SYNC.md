# 🔄 Guía de Sincronización Git - Aprendizaje IA

## 📍 Estado Actual

✅ Repositorio creado: `https://github.com/GabrielF2311/aprendizaje-ia`
✅ Todo sincronizado desde tu PC principal
✅ Listo para trabajar desde múltiples dispositivos

---

## 💻 Configurar en tu Laptop (Primera Vez)

### 1. Clonar el Repositorio

```powershell
# Navega a donde quieras tener el proyecto
cd "C:\Users\TU_USUARIO\Documents"

# Clona el repositorio
git clone https://github.com/GabrielF2311/aprendizaje-ia.git

# Entra al directorio
cd aprendizaje-ia
```

### 2. Configurar Git (si es la primera vez)

```powershell
git config --global user.name "Tu Nombre"
git config --global user.email "tu_email@ejemplo.com"
```

### 3. Configurar el Entorno Python

```powershell
# Crea el entorno virtual
python -m venv venv

# Activa el entorno
.\venv\Scripts\Activate.ps1

# Instala las dependencias
pip install -r requirements.txt

# Verifica la instalación
python verifica_instalacion.py
```

---

## 🔄 Workflow Diario

### Cuando Empieces a Trabajar

**SIEMPRE** haz pull primero para obtener los últimos cambios:

```powershell
# 1. Activa el entorno virtual
.\venv\Scripts\Activate.ps1

# 2. Obtén los últimos cambios
git pull origin main

# 3. Ahora trabaja normalmente
```

### Cuando Termines de Trabajar

Sube tus cambios al repositorio:

```powershell
# 1. Ver qué archivos cambiaron
git status

# 2. Agregar todos los cambios
git add -A

# O agregar archivos específicos
git add archivo1.py archivo2.md

# 3. Hacer commit con mensaje descriptivo
git commit -m "feat: Completé ejercicios de álgebra lineal día 1"

# 4. Subir a GitHub
git push origin main
```

---

## 💡 Mensajes de Commit Recomendados

Usa prefijos para mantener claridad:

```bash
# Cuando completes ejercicios
git commit -m "feat: Completé ejercicios NumPy día 3"

# Cuando completes un proyecto
git commit -m "feat: Proyecto transformaciones 2D terminado"

# Cuando agregues notas o teoría
git commit -m "docs: Añadí notas sobre backpropagation"

# Cuando corrijas errores
git commit -m "fix: Corregí error en función magnitude()"

# Cuando actualices progreso
git commit -m "chore: Actualicé MI_PROGRESO.md semana 5"
```

---

## 🚨 Solución de Problemas Comunes

### Conflictos de Merge

Si trabajaste en ambos dispositivos sin sincronizar:

```powershell
# Intenta pull
git pull origin main

# Si hay conflictos, verás algo como:
# CONFLICT (content): Merge conflict in archivo.py

# Abre el archivo en VS Code y resuelve manualmente
# Busca las líneas con <<<<<<, ======, >>>>>>

# Después de resolver:
git add archivo.py
git commit -m "Merge: Resuelto conflicto en archivo.py"
git push origin main
```

### Descartar Cambios Locales

Si quieres eliminar cambios que hiciste:

```powershell
# Descartar cambios de un archivo específico
git checkout -- archivo.py

# Descartar TODOS los cambios (¡cuidado!)
git reset --hard origin/main
```

### Ver Historial

```powershell
# Ver commits recientes
git log --oneline -10

# Ver cambios en un archivo
git log -p archivo.py
```

---

## 📂 Estructura Recomendada

```
PC Principal:
└── C:\Users\gmfe2\OneDrive\Documentos\Código\IA\

Laptop:
└── C:\Users\TU_USUARIO\Documents\aprendizaje-ia\
```

Ambos apuntan al mismo repositorio de GitHub.

---

## ✅ Checklist Diario

### Antes de Empezar
- [ ] Activar entorno virtual
- [ ] `git pull origin main`
- [ ] Verificar que todo está actualizado

### Al Terminar
- [ ] `git status` (revisar cambios)
- [ ] `git add -A` (agregar cambios)
- [ ] `git commit -m "mensaje descriptivo"`
- [ ] `git push origin main`
- [ ] Verificar en GitHub que se subió

---

## 🎯 Comandos Rápidos (Cheat Sheet)

```powershell
# VER ESTADO
git status                    # Ver archivos modificados
git log --oneline -5         # Ver últimos 5 commits

# SINCRONIZAR
git pull origin main         # Bajar cambios
git push origin main         # Subir cambios

# HACER CAMBIOS
git add archivo.py           # Agregar archivo específico
git add -A                   # Agregar todos
git commit -m "mensaje"      # Commit
git push                     # Subir (si ya hiciste -u antes)

# DESHACER
git checkout -- archivo.py   # Descartar cambios en archivo
git reset HEAD archivo.py    # Quitar archivo del staging
git revert <commit-hash>     # Revertir un commit

# BRANCHES (AVANZADO)
git branch                   # Ver branches
git checkout -b feature      # Crear y cambiar a branch
git merge feature            # Fusionar branch
```

---

## 📱 GitHub en tu Teléfono

Puedes ver tu progreso desde cualquier lugar:

1. Ve a: `https://github.com/GabrielF2311/aprendizaje-ia`
2. Navega por los archivos
3. Lee teoría desde tu teléfono
4. Revisa tus commits

---

## 🔐 Autenticación

GitHub usa tokens en lugar de contraseñas:

Si te pide autenticación:
1. Se abrirá el navegador
2. Inicia sesión en GitHub
3. Autoriza la aplicación
4. Ya quedará configurado

---

## 💡 Tips Pro

1. **Commits pequeños y frecuentes**: Mejor 5 commits pequeños que 1 gigante
2. **Pull antes de push**: Evita conflictos
3. **Mensajes claros**: Tu yo del futuro te lo agradecerá
4. **No subas archivos grandes**: Ya está en .gitignore
5. **Revisa en GitHub**: Verifica que se subió correctamente

---

## 🆘 Si Algo Sale Mal

### Opción 1: Guardar cambios y empezar de nuevo

```powershell
# Guarda tus cambios en algún lado
cp -r . ../backup-ia

# Elimina el repo local
cd ..
rm -rf aprendizaje-ia

# Clona de nuevo
git clone https://github.com/GabrielF2311/aprendizaje-ia.git
```

### Opción 2: Contactar para ayuda

- Pregúntame si algo no funciona
- Revisa la documentación: https://git-scm.com/doc

---

## ✨ Ventajas de Esta Configuración

✅ **Trabaja desde cualquier lugar**: PC, laptop, universidad
✅ **Nunca pierdas tu progreso**: Todo en la nube
✅ **Historial completo**: Ve cómo has evolucionado
✅ **Portfolio**: Tu repo de GitHub muestra tu aprendizaje
✅ **Backup automático**: Protección contra fallos de disco

---

## 📅 Próximos Pasos

1. **En tu laptop**: Clona el repositorio y configura el entorno
2. **Prueba el workflow**: Haz un cambio pequeño, commit y push
3. **Verifica en PC**: Haz pull y verifica que el cambio llegó
4. **Repite**: Mantén sincronizado siempre

---

**¡Listo! Ahora puedes trabajar desde cualquier dispositivo! 🚀**

Recuerda: **Pull antes de trabajar, Push al terminar**
