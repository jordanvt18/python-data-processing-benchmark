# Python Data Processing Libraries Benchmark
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene una comparativa detallada de rendimiento entre las principales librerías de procesamiento de datos en Python: Pandas, Polars y Data.table.

## 🚀 Características

- Benchmarks detallados de operaciones comunes
- Casos de uso reales
- Comparativas de memoria y tiempo de ejecución
- Ejemplos de código optimizado para cada librería

## 📋 Requisitos

- Python 3.8+
- pandas
- polars
- datatable
- numpy
- jupyter

## 🔧 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/python-data-processing-benchmark.git
cd python-data-processing-benchmark
```

2. Crea un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📊 Estructura del Proyecto

```
python-data-processing-benchmark/
├── notebooks/
│   ├── 01_basic_operations.ipynb
│   ├── 02_advanced_operations.ipynb
│   └── 03_memory_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── benchmarks.py
│   └── utils.py
├── data/
│   └── sample_data.csv
├── tests/
│   └── test_benchmarks.py
├── requirements.txt
├── setup.py
└── README.md
```

## 💻 Uso

1. Inicia Jupyter Notebook:
```bash
jupyter notebook
```

2. Abre los notebooks en la carpeta `notebooks/` para ver las comparativas detalladas.

## 📈 Resultados

Los resultados detallados de los benchmarks se encuentran en los notebooks, pero aquí hay un resumen:

- Pandas: Mejor para prototipado rápido y datasets pequeños
- Polars: Rendimiento superior en operaciones complejas y grandes datasets
- Data.table: Excelente balance entre velocidad y facilidad de uso

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor, lee las guías de contribución antes de enviar un Pull Request.

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
