"""
Funciones de utilidad para el proyecto BCI Benchmark
"""

import random
import numpy as np
import torch
import os
import logging
import yaml
from datetime import datetime
import pandas as pd


def set_seed(seed=42):
    """
    Establece la semilla para reproducibilidad
    
    Parameters:
    -----------
    seed : int
        Valor de la semilla
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Semilla establecida en: {seed}")


def setup_logging(log_dir='results', log_level=logging.INFO):
    """
    Configura el sistema de logging
    
    Parameters:
    -----------
    log_dir : str
        Directorio donde guardar los logs
    log_level : int
        Nivel de logging (logging.INFO, logging.DEBUG, etc.)
        
    Returns:
    --------
    logger : logging.Logger
        Logger configurado
    """
    # Crear directorio de logs si no existe
    os.makedirs(log_dir, exist_ok=True)
    
    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'benchmark_{timestamp}.log')
    
    # Configurar formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurar logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado. Archivo: {log_file}")
    
    return logger


def load_config(config_path):
    """
    Carga un archivo de configuración YAML
    
    Parameters:
    -----------
    config_path : str
        Ruta al archivo de configuración
        
    Returns:
    --------
    config : dict
        Diccionario con la configuración
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results, output_dir='results', filename='results.csv'):
    """
    Guarda los resultados en un archivo CSV
    
    Parameters:
    -----------
    results : list of dict or pd.DataFrame
        Resultados a guardar
    output_dir : str
        Directorio de salida
    filename : str
        Nombre del archivo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertir a DataFrame si es necesario
    if isinstance(results, list):
        df = pd.DataFrame(results)
    elif isinstance(results, pd.DataFrame):
        df = results
    else:
        raise ValueError("results debe ser una lista de diccionarios o un DataFrame")
    
    # Guardar CSV
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Resultados guardados en: {filepath}")
    
    return filepath


def save_model(model, output_dir='results', filename=None):
    """
    Guarda un modelo entrenado
    
    Parameters:
    -----------
    model : torch.nn.Module
        Modelo a guardar
    output_dir : str
        Directorio de salida
    filename : str, optional
        Nombre del archivo. Si es None, se genera automáticamente con timestamp
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_{timestamp}.pth'
    
    filepath = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Modelo guardado en: {filepath}")
    
    return filepath


def get_device():
    """
    Obtiene el dispositivo disponible (CUDA o CPU)
    
    Returns:
    --------
    device : torch.device
        Dispositivo a usar
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Usando CPU")
    
    return device

