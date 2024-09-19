# Librería de análisis de datos en Python

## Introducción
Esta librería es una colección de módulos, funciones y clases diseñados para facilitar el análisis de datos en Python. Incluye herramientas para:

- Manejo de valores faltantes y outliers.
- Perfilamiento y generación de reportes de datos.
- Cálculos estadísticos y pruebas de hipótesis.
- Funciones utilitarias generales.
- Generación de visualizaciones de datos.

## Contenido

- **NanHandler**: Clase para gestionar valores faltantes en conjuntos de datos.
- **OutlierHandler**: Clase para detectar y tratar outliers utilizando diferentes métodos estadísticos.
- **DataProfiler**: Clase para generar reportes detallados de perfilamiento de datos en formato HTML.
- **statistics_utils.py**: Módulo con funciones estadísticas comunes, como medidas descriptivas, pruebas de hipótesis, análisis de correlación y más.

## Requisitos

- Python 3.x
- Bibliotecas:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - seaborn

## Instalación

Actualmente, la librería se puede instalar clonando el repositorio y agregándola al PYTHONPATH:

```bash
git clone https://github.com/joferte84/profiler_library.git
```

## Primeros Pasos
Importa los módulos o clases que necesites en tu script:

```bash
from nans import NanHandler
from outliers import OutlierHandler
from profiler import DataProfiler
from statistics_utils import calculate_mean
from utils import read_file
from visualizations import Visualizer
```

Pendiente de agregar ejemplos y documentación de uso.
