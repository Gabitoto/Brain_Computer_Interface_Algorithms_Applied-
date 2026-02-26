"""
Script simplificado para entrenar EEGNet con PyTorch
1. Descarga/prepara datos
2. Entrena EEGNet
3. Evalúa y guarda resultados
"""

import argparse
import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery

# Importar utilidades y modelo
from src.utils import set_seed, setup_logging, load_config, get_device, save_model
from models.custom_models import EEGNet


def load_data(dataset_name="BNCI2014001", data_dir="data", n_subjects=None):
    """
    Descarga y carga el dataset
    
    Parameters:
    -----------
    dataset_name : str
        Nombre del dataset
    data_dir : str
        Directorio donde guardar los datos
    n_subjects : int, optional
        Número de sujetos a usar (None para todos)
        
    Returns:
    --------
    X_train, y_train, X_test, y_test : arrays
        Datos de entrenamiento y prueba
    """
    print(f"Descargando dataset {dataset_name}...")
    
    # Crear directorio de datos si no existe
    os.makedirs(data_dir, exist_ok=True)
    
    # Cargar dataset
    dataset = BNCI2014001()
    paradigm = MotorImagery()
    
    # Obtener sujetos
    if n_subjects:
        subjects = list(range(1, n_subjects + 1))
    else:
        subjects = None
    
    # Obtener datos
    X, y, metadata = paradigm.get_data(dataset, subjects=subjects)
    
    print(f"Datos cargados: {X.shape}, Etiquetas: {y.shape}")
    print(f"Clases: {np.unique(y)}")
    
    # Dividir en entrenamiento y prueba (usar sesiones)
    # Por simplicidad, usar la primera sesión para entrenar y la segunda para probar
    train_idx = metadata['session'] == metadata['session'].unique()[0]
    test_idx = metadata['session'] == metadata['session'].unique()[1]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def train_model(model, train_loader, val_loader, config, device, logger):
    """
    Entrena el modelo EEGNet
    
    Parameters:
    -----------
    model : nn.Module
        Modelo a entrenar
    train_loader : DataLoader
        DataLoader de entrenamiento
    val_loader : DataLoader
        DataLoader de validación
    config : dict
        Configuración de entrenamiento
    device : torch.device
        Dispositivo a usar
    logger : logging.Logger
        Logger para mensajes
        
    Returns:
    --------
    model : nn.Module
        Modelo entrenado
    history : dict
        Historial de entrenamiento
    """
    training_config = config['training']
    
    # Configurar optimizador y pérdida
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.0001)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Historial
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    n_epochs = training_config['n_epochs']
    
    logger.info(f"Iniciando entrenamiento por {n_epochs} épocas...")
    
    for epoch in range(n_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calcular promedios
        train_loss_avg = train_loss / len(train_loader)
        train_acc_avg = 100 * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc_avg = 100 * val_correct / val_total
        
        # Guardar historial
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc_avg)
        
        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Época [{epoch+1}/{n_epochs}] - "
                f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.2f}% - "
                f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.2f}%"
            )
    
    return model, history


def evaluate_model(model, test_loader, device, logger):
    """
    Evalúa el modelo en datos de prueba
    
    Parameters:
    -----------
    model : nn.Module
        Modelo a evaluar
    test_loader : DataLoader
        DataLoader de prueba
    device : torch.device
        Dispositivo a usar
    logger : logging.Logger
        Logger para mensajes
        
    Returns:
    --------
    accuracy : float
        Precisión del modelo
    y_true, y_pred : arrays
        Etiquetas verdaderas y predicciones
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"\nPrecisión en prueba: {accuracy:.4f}")
    logger.info(f"\nReporte de clasificación:\n{classification_report(all_labels, all_preds)}")
    
    return accuracy, np.array(all_labels), np.array(all_preds)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Entrenar EEGNet con PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Entrenar con configuración por defecto
  python benchmark.py
  
  # Entrenar con configuración personalizada
  python benchmark.py --config configs/eegnet.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eegnet.yaml',
        help='Ruta al archivo de configuración'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directorio donde guardar los datos'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directorio donde guardar los resultados'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logger = setup_logging(args.output_dir)
    logger.info("Iniciando entrenamiento de EEGNet")
    
    # Establecer semilla
    set_seed(42)
    
    # Obtener dispositivo
    device = get_device()
    
    # Cargar configuración
    logger.info(f"Cargando configuración: {args.config}")
    config = load_config(args.config)
    
    # Cargar datos
    logger.info("Cargando datos...")
    n_subjects = config.get('data', {}).get('n_subjects', None)
    X_train, y_train, X_test, y_test = load_data(
        data_dir=args.data_dir,
        n_subjects=n_subjects
    )
    
    # Normalizar datos
    logger.info("Normalizando datos...")
    scaler = StandardScaler()
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_chans = X_train.shape[1]
    n_times = X_train.shape[2]
    
    # Reshape para normalización: (n_samples, n_features)
    X_train_2d = X_train.reshape(n_train, -1)
    X_test_2d = X_test.reshape(n_test, -1)
    
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, n_chans, n_times)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, n_chans, n_times)
    
    # Convertir a tensores de PyTorch
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Crear datasets y dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Usar parte del entrenamiento para validación
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # Crear modelo
    model_config = config['model']
    filters_config = config.get('filters', {})
    
    logger.info("Creando modelo EEGNet...")
    model = EEGNet(
        n_classes=model_config['n_classes'],
        n_chans=model_config['n_chans'],
        n_times=model_config['n_times'],
        sfreq=model_config['sfreq'],
        **filters_config
    ).to(device)
    
    logger.info(f"Modelo creado. Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entrenar modelo
    model, history = train_model(model, train_loader, val_loader, config, device, logger)
    
    # Evaluar modelo
    logger.info("Evaluando modelo en datos de prueba...")
    accuracy, y_true, y_pred = evaluate_model(model, test_loader, device, logger)
    
    # Guardar modelo
    logger.info("Guardando modelo...")
    save_model(model, args.output_dir, 'eegnet_model.pth')
    
    # Guardar historial
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_file = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_file, index=False)
    logger.info(f"Historial guardado en: {history_file}")
    
    logger.info("\n¡Entrenamiento completado!")
    logger.info(f"Precisión final: {accuracy:.4f}")
    logger.info(f"Resultados guardados en: {args.output_dir}")


if __name__ == '__main__':
    main()