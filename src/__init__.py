"""
Python Data Processing Libraries Benchmark
=======================================

Este paquete proporciona herramientas para comparar el rendimiento de
las principales bibliotecas de procesamiento de datos en Python:
Pandas, Polars y Data.table.

Módulos
-------
benchmarks : Funciones para realizar pruebas de rendimiento
utils : Utilidades comunes para medición y visualización
"""

__version__ = '0.1.0'
__author__ = 'Jordan'


# Importar funciones principales para facilitar el acceso
from .benchmarks import (
    create_sample_data,
    run_benchmark,
    BenchmarkRunner
)

from .utils import (
    plot_benchmark_results,
    calculate_speedup,
    prepare_data_structure,
    get_memory_usage,
    memory_usage_report,
    generate_performance_report,
    save_benchmark_results,
    profile_memory_growth
)

# Definir qué módulos se expondrán con "from package import *"
__all__ = [
    'create_sample_data',
    'run_benchmark',
    'BenchmarkRunner',
    'plot_benchmark_results',
    'calculate_speedup',
    'prepare_data_structure',
    'get_memory_usage',
    'memory_usage_report',
    'generate_performance_report',
    'save_benchmark_results',
    'profile_memory_growth'
]