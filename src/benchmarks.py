"""
Main module to run benchmarks for data processing libraries.
"""
import pandas as pd
import polars as pl
import datatable as dt
import numpy as np
from time import time
import os
from typing import Dict, Any, List, Tuple
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_rows: int) -> Dict[str, Any]:
    """
    Crear conjunto de datos de ejemplo para benchmarks.
    
    Args:
        n_rows: Number of rows for the dataset
        
    Returns:
        Dictionary with sample data
    """
    rng = np.random.RandomState(42)
    return {
        'id': np.arange(n_rows),
        'value': rng.randn(n_rows),
        'category': rng.choice(['A', 'B', 'C'], n_rows),
        'date': pd.date_range('2000-01-01', periods=n_rows, freq='s') # Cambiar la frecuencia a segundos
    }

class BenchmarkRunner:
    """Class to run and manage benchmarks."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.results: Dict[str, Dict[str, float]] = {}
        
    def run_io_benchmark(self) -> Dict[str, float]:
        """Run IO operations benchmark."""
        results = {}
        
        # Pandas benchmark
        try:
            start = time()
            df = pd.DataFrame(self.data)
            df.to_csv('test_pandas.csv', index=False)
            df_read = pd.read_csv('test_pandas.csv')
            results['pandas'] = time() - start
        except Exception as e:
            logger.error(f"Error in Pandas benchmark: {e}")
            results['pandas'] = float('nan')
        finally:
            if os.path.exists('test_pandas.csv'):
                os.remove('test_pandas.csv')
            
        # Polars benchmark
        try:
            start = time()
            df = pl.DataFrame(self.data)
            df.write_csv('test_polars.csv')
            df_read = pl.read_csv('test_polars.csv')
            results['polars'] = time() - start
        except Exception as e:
            logger.error(f"Error in Polars benchmark: {e}")
            results['polars'] = float('nan')
        finally:
            if os.path.exists('test_polars.csv'):
                os.remove('test_polars.csv')
            
        # Data.table benchmark
        try:
            start = time()
            data_copy = self.data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date']).astype(np.int64)  # Convertir a int64
            # Convertir las columnas a listas para evitar el error
            df = dt.Frame({k: list(v) for k, v in data_copy.items()})
            df.to_csv('test_datatable.csv')
            df_read = dt.fread('test_datatable.csv')
            results['datatable'] = time() - start
        except Exception as e:
            logger.error(f"Error in Data.table benchmark: {e}")
            results['datatable'] = float('nan')
        finally:
            if os.path.exists('test_datatable.csv'):
                os.remove('test_datatable.csv')
            
        return results
    
    # Corrección en el método run_groupby_benchmark
    def run_groupby_benchmark(self) -> Dict[str, float]:
        """Run benchmark for groupby operations."""
        results = {}
        
        # Pandas benchmark
        try:
            start = time()
            df = pd.DataFrame(self.data)
            df.groupby('category').agg({'value': 'mean'})
            results['pandas'] = time() - start
        except Exception as e:
            logger.error(f"Error in Pandas groupby benchmark: {e}")
            results['pandas'] = float('nan')
        
        # Polars benchmark
        try:
            start = time()
            df = pl.DataFrame(self.data)
            df.group_by('category').agg(pl.col('value').mean())
            results['polars'] = time() - start
        except Exception as e:
            logger.error(f"Error in Polars groupby benchmark: {e}")
            results['polars'] = float('nan')
        
            # Data.table benchmark
        # Data.table benchmark
        try:
            start = time()
            # Convertir las columnas a listas y manejar las fechas
            data_copy = self.data.copy()
            data_copy['date'] = data_copy['date'].astype(np.int64)  # Convertir a int64
            df = dt.Frame({k: list(v) for k, v in data_copy.items()})  # Convertir a listas
            df[:, dt.mean(dt.f.value), dt.by(dt.f.category)]
            results['datatable'] = time() - start
        except Exception as e:
            logger.error(f"Error in Data.table groupby benchmark: {e}")
            results['datatable'] = float('nan')
        
        return results
    
    def run_all_benchmarks(self, operations: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Run specified benchmarks or all if none specified."""
        if operations is None:
            operations = ['io', 'groupby']
        
        if 'io' in operations:
            self.results['io'] = self.run_io_benchmark()
        if 'groupby' in operations:
            self.results['groupby'] = self.run_groupby_benchmark()
        
        return self.results

def run_benchmark(data: Dict[str, Any], operations: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Main function to run benchmarks.
    
    Args:
        data: Data to use in the benchmarks
        operations: List of operations to benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    runner = BenchmarkRunner(data)
    return runner.run_all_benchmarks(operations)

if __name__ == "__main__":
    # Ejemplo de uso
    test_data = create_sample_data(1_000_000)
    results = run_benchmark(test_data)
    print("Resultados de los benchmarks:", results)