"""
Utilidades para el proyecto de benchmarking de librerías de procesamiento de datos.
Incluye funciones para preparación de datos, visualización y análisis de rendimiento.
"""
import pandas as pd
import polars as pl
import datatable as dt
import numpy as np
from typing import Dict, Any, Union, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import psutil
import os
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data_structure(data: Dict[str, Any]) -> Dict[str, Union[pd.DataFrame, pl.DataFrame, dt.Frame]]:
    """
    Prepara los datos en el formato correcto para cada librería.
    
    Args:
        data: Diccionario con los datos de entrada
        
    Returns:
        Diccionario con DataFrames para cada librería
    """
    try:
        return {
            'pandas': pd.DataFrame(data),
            'polars': pl.DataFrame(data),
            'datatable': dt.Frame(data)
        }
    except Exception as e:
        logger.error(f"Error preparando estructuras de datos: {e}")
        raise

def get_memory_usage(obj: Any) -> float:
    """
    Calcula el uso de memoria de un objeto en MB.
    
    Args:
        obj: Objeto a medir
        
    Returns:
        Uso de memoria en MB
    """
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1024**2
    elif isinstance(obj, pl.DataFrame):
        return obj.estimated_size() / 1024**2
    elif isinstance(obj, dt.Frame):
        return obj.memory_footprint() / 1024**2
    else:
        return 0

def memory_usage_report(data_frames: Dict[str, Union[pd.DataFrame, pl.DataFrame, dt.Frame]]) -> pd.DataFrame:
    """
    Genera un reporte detallado del uso de memoria para cada implementación.
    
    Args:
        data_frames: Diccionario con DataFrames de cada librería
        
    Returns:
        DataFrame con el uso de memoria y métricas relacionadas
    """
    memory_stats = {}
    process = psutil.Process(os.getpid())
    
    for name, df in data_frames.items():
        initial_memory = process.memory_info().rss / 1024**2
        df_memory = get_memory_usage(df)
        
        memory_stats[name] = {
            'DataFrame Size (MB)': df_memory,
            'Total Process Memory (MB)': initial_memory,
            'Rows': len(df),
            'Memory per Row (KB)': (df_memory * 1024) / len(df)
        }
    
    return pd.DataFrame(memory_stats).T

def plot_benchmark_results(results: Dict[str, Dict[str, float]], 
                         title: str = "Benchmark Results",
                         save_path: Optional[Union[str, Path]] = None,
                         plot_type: str = 'bar') -> None:
    """
    Visualiza los resultados del benchmark con diferentes tipos de gráficos.
    
    Args:
        results: Resultados del benchmark
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
        plot_type: Tipo de gráfico ('bar', 'box', 'violin')
    """
    plt.figure(figsize=(12, 6))
    df_results = pd.DataFrame(results).T
    
    if plot_type == 'bar':
        sns.barplot(data=df_results)
    elif plot_type == 'box':
        sns.boxplot(data=df_results)
    elif plot_type == 'violin':
        sns.violinplot(data=df_results)
    else:
        raise ValueError(f"Tipo de gráfico no soportado: {plot_type}")
    
    plt.title(title)
    plt.ylabel("Tiempo (segundos)")
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def generate_performance_report(benchmark_results: Dict[str, Dict[str, float]], 
                              memory_results: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un reporte completo de rendimiento combinando tiempo y memoria.
    
    Args:
        benchmark_results: Resultados de los benchmarks de tiempo
        memory_results: Resultados del análisis de memoria
        
    Returns:
        DataFrame con el reporte completo
    """
    # Convertir resultados de benchmark a DataFrame
    time_df = pd.DataFrame(benchmark_results).T
    
    # Combinar con resultados de memoria
    report = pd.merge(
        time_df, 
        memory_results, 
        left_index=True, 
        right_index=True,
        suffixes=('_time', '_memory')
    )
    
    # Calcular métricas adicionales
    report['time_memory_ratio'] = report['DataFrame Size (MB)'] / report.iloc[:, 0]
    
    return report

def save_benchmark_results(results: Dict[str, Any], 
                         filename: str = None) -> None:
    """
    Guarda los resultados del benchmark en un archivo CSV.
    
    Args:
        results: Resultados a guardar
        filename: Nombre del archivo (opcional)
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'benchmark_results_{timestamp}.csv'
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    pd.DataFrame(results).to_csv(output_dir / filename)
    logger.info(f"Resultados guardados en: {output_dir / filename}")

def calculate_speedup(results: Dict[str, float], 
                     baseline: str = 'pandas') -> Dict[str, float]:
    """
    Calcula la mejora de velocidad relativa respecto a una línea base.
    
    Args:
        results: Diccionario con tiempos de ejecución
        baseline: Librería a usar como referencia
        
    Returns:
        Diccionario con los speedups calculados
    """
    baseline_time = results[baseline]
    return {
        lib: baseline_time / time 
        for lib, time in results.items()
    }

def profile_memory_growth(func: callable, 
                         iterations: int = 5) -> List[float]:
    """
    Perfila el crecimiento de memoria durante la ejecución repetida.
    
    Args:
        func: Función a perfilar
        iterations: Número de iteraciones
        
    Returns:
        Lista con el uso de memoria en cada iteración
    """
    process = psutil.Process(os.getpid())
    memory_usage = []
    
    for _ in range(iterations):
        initial_memory = process.memory_info().rss / 1024**2
        func()
        final_memory = process.memory_info().rss / 1024**2
        memory_usage.append(final_memory - initial_memory)
    
    return memory_usage

# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de prueba
    test_data = {
        'A': range(1000),
        'B': np.random.randn(1000)
    }
    
    # Preparar estructuras de datos
    dfs = prepare_data_structure(test_data)
    
    # Generar reporte de memoria
    memory_report = memory_usage_report(dfs)
    print("\nReporte de Memoria:")
    print(memory_report)
