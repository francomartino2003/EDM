# Análisis de Datos EDM - Resumen Completo

## 📊 Descripción del Proyecto

Este proyecto contiene un análisis exhaustivo de datos de perforación EDM (Electrical Discharge Machining) organizados en dos opciones principales con datos de entrenamiento y prueba.

## 🗂️ Estructura de Datos

### Datos Originales
```
Data/
├── Option 1/
│   ├── Train/ (392 archivos)
│   │   ├── Normal/ (220 archivos)
│   │   ├── NPT/ (66 archivos)
│   │   ├── OD/ (75 archivos)
│   │   ├── AbnD/ (18 archivos)
│   │   ├── MH/ (6 archivos)
│   │   ├── PNormal/ (2 archivos)
│   │   └── POD/ (5 archivos)
│   └── Test/ (390 archivos)
│       └── [mismas categorías]
└── Option 2/
    ├── Train/ (102 archivos)
    └── Test/ (101 archivos)
        └── [categorías: Normal, NPT, OD, etc.]
```

### Visualizaciones Generadas
```
Data Viz/
├── Option 1/ (827 visualizaciones - SIN colores de stages)
└── Option 2/ (203 visualizaciones - CON colores de stages)
```

## 🔍 Diferencias Clave entre Option 1 y Option 2

| Característica | Option 1 | Option 2 |
|---|---|---|
| **Total archivos** | 782 (Train: 392, Test: 390) | 203 (Train: 102, Test: 101) |
| **Columna Stage** | ❌ No tiene | ✅ Tiene (Segment) |
| **Colores en gráficos** | Solo voltaje y Z | Voltaje, Z + colores por stage |
| **Relación** | Dataset completo | Subset filtrado de Option 1 |

## 📈 Análisis de Free Falling

### 🏆 Free Falling Consecutivo Más Largo
- **Archivo ganador**: `8e1f46815d8d4169b6e8a83a1546e159` (Normal)
- **Longitud**: **2,191 muestras consecutivas**
- **Porcentaje**: 39.1% del archivo completo
- **Muestras**: 587 a 2,777
- **Especialidad**: Es el único archivo de Normal que contiene "Rework Free Falling"

### Top 3 Free Falling Más Largos (Global)
1. **Normal**: 2,191 muestras (39.1%) - `8e1f46815d8d4169b6e8a83a1546e159`
2. **OD**: 1,340 muestras (12.7%) - `1a47e7adc7ab4c04b360a12e9da25a28`
3. **OD**: 1,266 muestras (18.0%) - `5a255906f8b24a5db5d4d01e1f441330`

### Estadísticas por Categoría (excluyendo archivo especial)

#### 🔵 NPT
- **Archivos con Free Falling**: 0 (0.0%)
- **MIN/MAX/PROMEDIO**: N/A

#### 🟠 NORMAL
- **Archivos con Free Falling**: 55/56 (98.2%)
- **MINIMO**: 88 muestras
- **MAXIMO**: 378 muestras  
- **PROMEDIO**: 198.5 muestras

#### 🟢 OD
- **Archivos con Free Falling**: 19/19 (100.0%)
- **MINIMO**: 221 muestras
- **MAXIMO**: 1,340 muestras
- **PROMEDIO**: 467.5 muestras

## 📏 Análisis de Longitud de Series

### Estadísticas por Categoría (excluyendo archivo especial)

| Categoría | Mínimo | Máximo | Promedio | Mediana |
|---|---|---|---|---|
| **NPT** | 990 | 9,257 | 3,100.8 | 2,773.0 |
| **Normal** | 666 | 6,257 | 2,834.1 | 2,629.0 |
| **OD** | 3,245 | 13,726 | 6,298.8 | 5,809.0 |

### Extremos Globales
- **Serie más corta**: 666 muestras (Normal)
- **Serie más larga**: 13,726 muestras (OD)
- **Diferencia**: 13,060 muestras

## 🔄 Análisis de Gaps en OD

### Diferencias entre Segmentos Free Falling (solo OD)
- **Total de gaps analizados**: 12
- **GAP MÍNIMO**: 778 muestras
- **GAP MÁXIMO**: 3,497 muestras
- **GAP PROMEDIO**: 1,778.8 muestras

### Top 5 Gaps Más Largos
1. 3,497 muestras - `52b0f155a8a640a6b5e2dc377753efc9`
2. 2,937 muestras - `16b39d6bab16475c97dd4d6e5802056a`
3. 2,813 muestras - `3a88bca38ff94c1eb4350a3625115f73`
4. 1,746 muestras - `0ba9bfda0b36424ab8fdc55b6c3da9a0`
5. 1,699 muestras - `3ca449515a094479a84fa1ad7c7f3cd9`

### Archivos con Múltiples Segmentos
- **1 archivo**: 3 segmentos de Free Falling
- **8 archivos**: 2 segmentos cada uno
- **10 archivos**: 1 segmento cada uno

## 🔗 Análisis de Archivos Compartidos

### Combinaciones de Archivos entre Opciones
- **Option 1/Train ↔ Option 2/Train**: 102 archivos compartidos
- **Option 1/Test ↔ Option 2/Test**: 101 archivos compartidos
- **Option 1/Train ↔ Option 1/Test**: 0 archivos (correcto para ML)
- **Total archivos únicos**: 782
- **Archivos compartidos**: 203 (26.0%)

### Conclusión
**Option 2 es un subset filtrado de Option 1**, representando un conjunto más pequeño y posiblemente más relevante de los datos originales.

## 🎨 Sistema de Colores para Stages

### Paleta de Colores (Option 2)
- **Touching**: `#dbeafe` (azul claro)
- **Pre-drilling**: `#e6f4ea` (verde claro)
- **Body Drilling**: `#fff4e5` (naranja claro)
- **Break-through**: `#fde8e8` (rosa claro)
- **Free Falling**: `#ff4444` (rojo fuerte) ⭐
- **Rework Free Falling**: `#8B0000` (rojo oscuro) ⭐
- **Retraction**: `#f3e8ff` (púrpura claro)
- **Scarfing**: `#e0f2fe` (cian claro)
- **Drilling**: `#f0f0f0` (gris claro)
- **Machine Health Issue**: `#ffcc00` (amarillo)

## 📊 Hallazgos Clave

### 1. Patrones de Free Falling
- **NPT nunca tiene Free Falling** - característica distintiva
- **Normal tiene Free Falling en 98.2%** de archivos
- **OD tiene Free Falling en 100%** de archivos con promedios más altos
- **El archivo especial** (`8e1f46815d8d4169b6e8a83a1546e159`) es único por su Free Falling extremadamente largo (2,191 muestras)

### 2. Complejidad de Procesos
- **OD representa procesos más largos** (promedio 6,298.8 muestras vs 2,834.1 en Normal)
- **OD tiene mayor variabilidad** en longitud de series
- **Gaps entre Free Falling en OD** son considerablemente largos (promedio 1,778.8 muestras)

### 3. Estructura de Datos
- **Option 2 es un subconjunto curado** de Option 1
- **No hay overlap entre Train y Test** (correcto para machine learning)
- **26% de los archivos** aparecen en múltiples opciones

## 🛠️ Herramientas y Scripts Utilizados

### Scripts Principales
1. `plot_voltage_z_by_stage.py` - Función de visualización principal
2. `find_longest_free_falling.py` - Análisis de Free Falling consecutivo
3. `analyze_csv_combinations.py` - Análisis de archivos compartidos
4. `create_data_viz_structure.py` - Generación de visualizaciones
5. `series_length_stats.py` - Estadísticas de longitud de series
6. `analyze_free_falling_gaps_od.py` - Análisis de gaps en OD

### Tecnologías
- **Python 3.x**
- **Pandas** - Manipulación de datos
- **Matplotlib** - Visualizaciones
- **NumPy** - Cálculos numéricos

## 📈 Visualizaciones Generadas

### Total de Gráficos Creados: 985
- **Option 1**: 827 visualizaciones (sin colores de stages)
- **Option 2**: 203 visualizaciones (con colores de stages)

### Características de los Gráficos
- **Voltaje**: Eje izquierdo (azul)
- **Z (profundidad)**: Eje derecho (naranja)
- **Stages**: Fondos coloreados (solo Option 2)
- **Resolución**: 300 DPI
- **Formato**: PNG

## 🎯 Conclusiones Generales

1. **Los datos muestran patrones claros** por categoría de proceso
2. **Free Falling es un indicador importante** de la fase de perforación
3. **OD representa procesos más complejos** y largos
4. **Option 2 proporciona un dataset más manejable** para análisis específicos
5. **El archivo especial** (`8e1f46815d8d4169b6e8a83a1546e159`) merece análisis adicional por sus características únicas

---

*Análisis completado con 985 visualizaciones generadas y múltiples métricas calculadas para caracterización completa del dataset EDM.*
