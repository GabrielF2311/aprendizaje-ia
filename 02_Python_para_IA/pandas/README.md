# 📊 Pandas para Datos - Semana 12

## 🎯 Objetivos de la Semana

- Dominar Series y DataFrames
- Manipulación eficiente de datos tabulares
- Limpieza de datos (missing values, duplicados)
- Agrupaciones y agregaciones
- Análisis exploratorio de datos (EDA)

## 🐼 ¿Qué es Pandas?

Pandas es **la** librería para manipulación de datos en Python:
- Construida sobre NumPy
- DataFrames similar a Excel/SQL/R
- Funciones para limpieza y transformación
- Integración perfecta con Matplotlib/Seaborn

---

## 📅 Plan de la Semana

### **Día 1: Series y DataFrames**
- Creación de Series
- Creación de DataFrames
- Atributos básicos (head, tail, info, describe)
- Lectura de archivos (CSV, Excel)

💻 **Ejercicios**: `ejercicios/dia_01_dataframes.py`

---

### **Día 2: Indexación y Selección**
- loc vs iloc
- Selección de columnas/filas
- Boolean indexing
- Query

💻 **Ejercicios**: `ejercicios/dia_02_seleccion.py`

---

### **Día 3: Limpieza de Datos**
- Detección de valores nulos
- Manejo de missing values (drop, fill, interpolate)
- Duplicados
- Conversión de tipos

💻 **Ejercicios**: `ejercicios/dia_03_limpieza.py`

---

### **Día 4: Transformaciones y Agregaciones**
- GroupBy
- Apply, Map, ApplyMap
- Merge, Join, Concat
- Pivot tables

💻 **Ejercicios**: `ejercicios/dia_04_agregacion.py`

---

### **Día 5-7: PROYECTO - Análisis Exploratorio de Datos**

Realiza un EDA completo de un dataset real:
- Carga y exploración inicial
- Limpieza de datos
- Visualizaciones
- Insights y conclusiones

📓 **Notebook**: `proyecto_eda.ipynb`

---

## 🔑 Conceptos Clave

### Series vs DataFrame

```python
import pandas as pd

# Series: 1D array con índice
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

# DataFrame: 2D tabla con filas y columnas
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos'],
    'edad': [25, 30, 35],
    'ciudad': ['Lima', 'Bogotá', 'CDMX']
})
```

### Selección: loc vs iloc

```python
# iloc: por posición (enteros)
df.iloc[0]        # Primera fila
df.iloc[0:2, 0:2] # Primeras 2 filas y columnas

# loc: por etiqueta (nombres)
df.loc[0, 'nombre']        # Valor específico
df.loc[df['edad'] > 28]    # Boolean indexing
```

---

## ✅ Checklist de Progreso

### Fundamentos
- [ ] Puedo crear Series y DataFrames
- [ ] Sé leer CSV/Excel
- [ ] Uso head(), info(), describe()
- [ ] Entiendo índices

### Selección de Datos
- [ ] Domino loc e iloc
- [ ] Uso boolean indexing
- [ ] Filtro datos con query()
- [ ] Selecciono múltiples columnas/filas

### Limpieza
- [ ] Detecto valores nulos
- [ ] Manejo missing values apropiadamente
- [ ] Elimino duplicados
- [ ] Convierto tipos de datos

### Transformaciones
- [ ] Uso GroupBy correctamente
- [ ] Aplico funciones con apply/map
- [ ] Combino DataFrames (merge, join, concat)
- [ ] Creo pivot tables

### Proyecto
- [ ] EDA completo de dataset real
- [ ] Visualizaciones informativas
- [ ] Insights documentados
- [ ] Código limpio y comentado

---

## 📊 Operaciones Comunes

### Exploración Inicial

```python
# Carga de datos
df = pd.read_csv('datos.csv')

# Exploración rápida
df.head()           # Primeras filas
df.tail()           # Últimas filas
df.info()           # Tipos y memoria
df.describe()       # Estadísticas
df.shape            # (filas, columnas)
df.columns          # Nombres de columnas
df.dtypes           # Tipos de datos
```

### Limpieza

```python
# Valores nulos
df.isnull().sum()           # Cuenta nulos por columna
df.dropna()                 # Elimina filas con nulos
df.fillna(0)                # Rellena nulos con 0
df['col'].fillna(df['col'].mean())  # Con media

# Duplicados
df.duplicated().sum()       # Cuenta duplicados
df.drop_duplicates()        # Elimina duplicados
```

### Agregaciones

```python
# GroupBy
df.groupby('categoria')['precio'].mean()
df.groupby(['ciudad', 'año'])['ventas'].sum()

# Pivot table
pd.pivot_table(df, 
               values='ventas',
               index='mes',
               columns='categoria',
               aggfunc='sum')
```

---

## 💡 Tips de la Semana

1. **Encadena operaciones**: `df.dropna().groupby('cat').mean()`
2. **Copia vs Vista**: Usa `.copy()` cuando modifiques
3. **Inplace**: Evita `inplace=True`, mejor asigna resultado
4. **Memory**: Usa `category` dtype para columnas con pocos valores únicos
5. **Performance**: `query()` es más rápido para filtros complejos

---

## 🎯 Mini-Desafíos

**Día 1**: Crea un DataFrame con tus películas favoritas
**Día 2**: Encuentra todas las filas donde una condición compleja se cumple
**Día 3**: Limpia un dataset "sucio" con múltiples problemas
**Día 4**: Calcula ventas por región y mes de un dataset

---

## 📚 Recursos

### Documentación
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

### Videos
- **Corey Schafer** - Pandas Tutorial (YouTube)
- **Data School** - Pandas Q&A

### Datasets para Practicar
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- Titanic, Housing, Sales, etc.

---

## 🚀 Siguiente Paso

Empieza con **Día 1**: `ejercicios/dia_01_dataframes.py`

**¡Domina Pandas esta semana!** 🐼
