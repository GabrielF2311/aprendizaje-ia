# 📊 Visualización de Datos - Semana 13

## 🎯 Objetivos de la Semana

- Crear gráficos con Matplotlib
- Visualizaciones estadísticas con Seaborn
- Gráficos interactivos con Plotly
- Diseñar dashboards informativos
- Comunicar insights visualmente

## 🎨 Por qué Visualización es Crucial

> "Una imagen vale más que mil palabras" - Especialmente en datos

- Detecta patrones y outliers instantáneamente
- Comunica resultados a no técnicos
- Valida modelos visualmente
- Explora datos antes de modelar

---

## 📅 Plan de la Semana

### **Día 1-2: Matplotlib Básico**

Domina la librería fundamental de visualización:
- Gráficos de líneas, barras, scatter
- Subplots y layouts
- Personalización (colores, estilos, anotaciones)
- Guardar figuras

📓 **Notebook**: `matplotlib_basico.ipynb`

---

### **Día 3-4: Seaborn - Visualización Estadística**

Gráficos hermosos y estadísticos:
- Distribuciones (histplot, kdeplot, boxplot)
- Relaciones (scatterplot, pairplot, heatmap)
- Categóricos (barplot, countplot, violinplot)
- Temas y estilos

📓 **Notebook**: `seaborn_estadistico.ipynb`

---

### **Día 5: Plotly - Gráficos Interactivos**

Visualizaciones modernas e interactivas:
- Gráficos básicos interactivos
- Hover tooltips
- Zoom, pan, select
- Exportar a HTML

📓 **Notebook**: `plotly_interactivo.ipynb`

---

### **Día 6-7: PROYECTO - Dashboard de Datos**

Crea un dashboard completo:
- Múltiples visualizaciones coordinadas
- Insights del dataset
- Diseño profesional
- Exportable/compartible

📓 **Notebook**: `proyecto_dashboard.ipynb`

---

## 🎨 Guía de Visualización

### ¿Qué gráfico usar?

| Objetivo | Tipo de Gráfico |
|----------|----------------|
| Comparar categorías | Barras, Columnas |
| Ver tendencias | Líneas |
| Mostrar distribución | Histograma, Box plot, Violin |
| Relaciones entre variables | Scatter plot, Pair plot |
| Proporciones | Pie chart, Donut |
| Correlaciones | Heatmap |
| Composición temporal | Stacked area, Stacked bars |

---

## 🖼️ Ejemplos de Código

### Matplotlib Básico

```python
import matplotlib.pyplot as plt
import numpy as np

# Gráfico de líneas
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Función Seno')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Seaborn - Múltiples Distribuciones

```python
import seaborn as sns

# Configurar estilo
sns.set_style('whitegrid')

# Box plot con categorías
sns.boxplot(data=df, x='categoria', y='precio')
plt.title('Distribución de Precios por Categoría')
plt.show()
```

### Plotly - Interactivo

```python
import plotly.express as px

# Scatter interactivo
fig = px.scatter(df, 
                 x='edad', 
                 y='salario',
                 color='departamento',
                 size='experiencia',
                 hover_data=['nombre'])
fig.show()
```

---

## ✅ Checklist de Progreso

### Matplotlib
- [ ] Creo gráficos de líneas, barras, scatter
- [ ] Uso subplots (múltiples gráficos)
- [ ] Personalizo colores, estilos, etiquetas
- [ ] Anoto puntos importantes
- [ ] Guardo figuras en alta calidad

### Seaborn
- [ ] Visualizo distribuciones con histplot/kdeplot
- [ ] Uso boxplot y violinplot
- [ ] Creo heatmaps de correlación
- [ ] Uso pairplot para exploración
- [ ] Aplico temas y estilos

### Plotly
- [ ] Creo gráficos interactivos básicos
- [ ] Personalizo hover tooltips
- [ ] Uso zoom y pan
- [ ] Exporto a HTML
- [ ] Creo gráficos 3D

### Proyecto Dashboard
- [ ] Diseñé layout profesional
- [ ] Incluí 5+ visualizaciones
- [ ] Documenté insights
- [ ] Código limpio y organizado

---

## 🎨 Principios de Diseño

### 1. Simplicidad
❌ No sobrecargues el gráfico
✅ Un mensaje principal por gráfico

### 2. Colores
❌ No uses más de 5 colores
✅ Usa paletas coherentes (ColorBrewer, Tableau)

### 3. Etiquetas
❌ No dejes ejes sin título
✅ Siempre etiqueta ejes y título

### 4. Leyendas
❌ No uses códigos crípticos
✅ Nombres descriptivos en leyenda

### 5. Escala
❌ No manipules escalas para exagerar
✅ Escalas honestas y claras

---

## 🎯 Ejemplos de Visualizaciones Efectivas

### Para Presentaciones

```python
# Estilo limpio para presentaciones
plt.style.use('seaborn-v0_8-talk')
sns.set_palette('Set2')

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df, x='mes', y='ventas', ax=ax)
ax.set_title('Ventas Mensuales 2024', fontsize=16, fontweight='bold')
ax.set_ylabel('Ventas ($M)', fontsize=12)
plt.tight_layout()
plt.savefig('ventas_2024.png', dpi=300, bbox_inches='tight')
```

### Para Exploración

```python
# Pairplot para ver todas las relaciones
sns.pairplot(df, hue='categoria', diag_kind='kde')
plt.show()
```

### Para Correlaciones

```python
# Heatmap de correlación
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, 
            annot=True, 
            cmap='coolwarm',
            center=0,
            square=True)
plt.title('Matriz de Correlación')
plt.show()
```

---

## 💡 Tips de la Semana

1. **Explora antes de presentar**: Muchos gráficos exploratorios → 1-2 para presentar
2. **Context matters**: Adapta visualización a la audiencia
3. **Color blind friendly**: Usa paletas accesibles
4. **Exporta en alta calidad**: `dpi=300` para publicaciones
5. **Anota lo importante**: Resalta insights con anotaciones

---

## 📚 Recursos

### Galerías de Inspiración
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Plotly Gallery](https://plotly.com/python/)

### Libros
- **"Storytelling with Data"** - Cole Nussbaumer Knaflic
- **"The Visual Display of Quantitative Information"** - Edward Tufte

### Herramientas
- [ColorBrewer](https://colorbrewer2.org/) - Paletas de colores
- [Coolors](https://coolors.co/) - Generador de paletas

---

## 🚀 Siguiente Paso

Abre `matplotlib_basico.ipynb` y empieza a crear gráficos!

**¡Visualiza tus datos esta semana!** 📊
