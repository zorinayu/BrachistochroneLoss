#!/usr/bin/env python3
"""
Test BRACHISTOCHRONE method on Electricity Load Diagrams dataset
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import zipfile

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, Adam, SGD
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import our modules
from src.models.mlp import MLPClassifier
from src.losses.brachistochrone import BrachistochroneLoss
from src.losses.brachistochrone_pro import BrachistochroneAdam, BrachistochroneSGD

def get_args():
    parser = argparse.ArgumentParser(description='Test BRACHISTOCHRONE on Electricity Load Diagrams dataset')
    parser.add_argument('--sample_size', type=int, default=2000, help='Sample size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='../outputs/electricity', help='Output directory')
    return parser.parse_args()

class ElectricityDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data/electricityloaddiagrams20112014.zip', sample_size=None, test_size=0.2, random_state=42):
        
        print("Loading Electricity Load Diagrams dataset...")
        
        # Try to load from zip file
        try:
            with zipfile.ZipFile(data_path, 'r') as zip_ref:
                # Extract to temp directory
                temp_dir = 'temp_electricity'
                zip_ref.extractall(temp_dir)
                
                # Look for CSV files
                csv_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                
                if csv_files:
                    # Load the first CSV file found
                    df = pd.read_csv(csv_files[0])
                    
                    # Handle missing values
                    df = df.dropna()
                    
                    # Try to identify target column (usually load or consumption)
                    target_cols = [col for col in df.columns if any(keyword in col.lower() 
                                 for keyword in ['load', 'consumption', 'demand', 'power', 'target', 'label'])]
                    
                    if target_cols:
                        target_col = target_cols[0]
                        X = df.drop(target_col, axis=1)
                        y = df[target_col]
                    else:
                        # Use last column as target
                        X = df.iloc[:, :-1]
                        y = df.iloc[:, -1]
                    
                    # Convert to numeric, handling non-numeric columns
                    X = X.apply(pd.to_numeric, errors='coerce')
                    y = pd.to_numeric(y, errors='coerce')
                    
                    # Remove rows with NaN values
                    mask = ~(X.isna().any(axis=1) | y.isna())
                    X = X[mask]
                    y = y[mask]
                    
                else:
                    raise FileNotFoundError("No CSV files found in zip")
                
                # Clean up temp directory
                import shutil
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            print(f"Failed to load from zip file: {e}")
            print("Using synthetic Electricity Load-like data")
            # Create synthetic electricity load data
            np.random.seed(random_state)
            n_samples = 2000
            n_features = 8
            
            # Generate time-based features (hour, day_of_week, month, etc.)
            hours = np.random.randint(0, 24, n_samples)
            days = np.random.randint(0, 7, n_samples)
            months = np.random.randint(1, 13, n_samples)
            temperatures = np.random.normal(20, 10, n_samples)  # Temperature effect
            
            # Create feature matrix
            X = np.column_stack([
                hours, days, months, temperatures,
                np.random.randn(n_samples, n_features - 4)
            ]).astype(np.float32)
            
            # Generate target (electricity load) based on features
            # Higher load during day hours, weekdays, summer/winter
            y = (np.sin(2 * np.pi * hours / 24) * 0.3 +  # Daily pattern
                 (days < 5).astype(float) * 0.2 +  # Weekday effect
                 np.sin(2 * np.pi * months / 12) * 0.1 +  # Seasonal pattern
                 np.abs(temperatures - 20) * 0.05 +  # Temperature effect
                 np.random.randn(n_samples) * 0.1).astype(np.float32)
            
            # Normalize target
            y = (y - y.mean()) / y.std()
        
        # Sample data if specified
        if sample_size and sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features and targets
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        
        self.n_features = X_train.shape[1]
        self.n_classes = 1  # Regression task
        
        print(f"Electricity Dataset: {len(self.X_train)} train, {len(self.X_test)} test")
        print(f"Features: {self.n_features}, Target: Continuous")

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

def train_model(model, train_loader, val_loader, epochs, device, brach_loss=None, optimizer_type='adamw'):
    """Train model with or without BRACHISTOCHRONE loss"""
    if optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=0.001)
    elif optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=0.001)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=0.001)
    else:
        optimizer = AdamW(model.parameters(), lr=0.001)
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            task_loss = criterion(logits, y)
            
            # Calculate BRACHISTOCHRONE loss
            if brach_loss is not None:
                # For regression, use flattened features
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                logits_flat = logits.view(batch_size, -1)
                
                if isinstance(brach_loss, (BrachistochroneAdam, BrachistochroneSGD)):
                    # Use improved version
                    h_list = [x_flat, logits_flat]
                    L_path, L_mono = brach_loss(h_list)
                    total_loss = task_loss + L_path + L_mono
                else:
                    # Use original version
                    h_list = [x_flat, logits_flat]
                    L_path, L_mono = brach_loss(h_list)
                    total_loss = task_loss + L_path + L_mono
            else:
                total_loss = task_loss
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
        
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}")
    
    total_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_time,
        'final_val_loss': val_losses[-1]
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    return mse, mae, r2

def test_electricity_dataset(sample_size, epochs, batch_size, device, output_dir):
    """Test all methods on Electricity dataset"""
    print("=" * 60)
    print("TESTING ELECTRICITY LOAD DIAGRAMS DATASET")
    print("=" * 60)
    
    # Load dataset
    dataset = ElectricityDataset(sample_size=sample_size)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(dataset.X_train, dataset.y_train)
    test_dataset = torch.utils.data.TensorDataset(dataset.X_test, dataset.y_test)
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define methods to test
    methods = {
        'Brachistochrone': BrachistochroneLoss(),
        'BrachistochroneAdam': BrachistochroneAdam(alpha=1.0, beta=0.1, gamma=0.1),
        'BrachistochroneSGD': BrachistochroneSGD(alpha=1.0, beta=0.1, gamma=0.1),
        'Adam': None,
        'SGD': None
    }
    
    results = {}
    
    for method_name, brach_loss in methods.items():
        print(f"\nTraining {method_name}...")
        
        # Determine epochs based on method
        if method_name == 'Brachistochrone':
            # Brachistochrone method: only 1 epoch
            method_epochs = 1
        elif method_name in ['BrachistochroneAdam', 'BrachistochroneSGD']:
            # Brachistochrone variants: 1, 2, 3 epochs
            method_epochs = [1, 2, 3]
        else:
            # Other methods: 1, 2, 3 epochs
            method_epochs = [1, 2, 3]
        
        if isinstance(method_epochs, list):
            # Multiple epochs for non-Brachistochrone methods
            for epoch_count in method_epochs:
                print(f"  Running {epoch_count} epoch(s)...")
                
                # Create model
                model = MLPRegressor(dataset.n_features, [64, 32], 1).to(device)
                
                # Determine optimizer type
                if method_name == 'BrachistochroneAdam':
                    optimizer_type = 'adam'  # Use Adam for BrachistochroneAdam
                elif method_name == 'BrachistochroneSGD':
                    optimizer_type = 'sgd'  # Use SGD for BrachistochroneSGD
                elif method_name == 'Adam':
                    optimizer_type = 'adam'
                elif method_name == 'SGD':
                    optimizer_type = 'sgd'
                else:
                    optimizer_type = 'adam'
                
                # Train model
                train_results = train_model(model, train_loader, val_loader, epoch_count, device, brach_loss, optimizer_type)
                
                # Evaluate on test set
                test_mse, test_mae, test_r2 = evaluate_model(model, test_loader, device)
                
                method_key = f"{method_name}_{epoch_count}epochs"
                results[method_key] = {
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'total_time': train_results['total_time'],
                    'final_val_loss': train_results['final_val_loss'],
                    'epochs': epoch_count
                }
                
                print(f"  Test MSE: {test_mse:.4f}")
                print(f"  Test MAE: {test_mae:.4f}")
                print(f"  Test R2: {test_r2:.4f}")
                print(f"  Total Time: {train_results['total_time']:.1f}s ({epoch_count} epochs)")
        else:
            # Single epoch for Brachistochrone methods
            # Create model
            model = MLPRegressor(dataset.n_features, [64, 32], 1).to(device)
            
            # Determine optimizer type
            optimizer_type = 'adamw'
            
            # Train model
            train_results = train_model(model, train_loader, val_loader, method_epochs, device, brach_loss, optimizer_type)
            
            # Evaluate on test set
            test_mse, test_mae, test_r2 = evaluate_model(model, test_loader, device)
            
            results[method_name] = {
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'total_time': train_results['total_time'],
                'final_val_loss': train_results['final_val_loss'],
                'epochs': method_epochs
            }
            
            print(f"  Test MSE: {test_mse:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  Test R2: {test_r2:.4f}")
            print(f"  Total Time: {train_results['total_time']:.1f}s ({method_epochs} epoch)")
    
    return results

def save_numerical_summary(results, output_dir):
    """Save numerical summary"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'electricity_results.txt'), 'w') as f:
        f.write("BRACHISTOCHRONE METHOD TEST RESULTS - ELECTRICITY LOAD DIAGRAMS DATASET\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"{'Method':<35} {'Test MSE':<10} {'Test MAE':<10} {'Test R2':<10} {'Time(s)':<15} {'Epochs':<10}\n")
        f.write("-" * 100 + "\n")
        
        for method, result in results.items():
            epochs_info = f"{result['epochs']} epoch{'s' if result['epochs'] > 1 else ''}"
            f.write(f"{method:<35} {result['test_mse']:<10.4f} "
                   f"{result['test_mae']:<10.4f} {result['test_r2']:<10.4f} "
                   f"{result['total_time']:<15.1f} {epochs_info:<10}\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        # Find best methods
        best_mse = min(results.keys(), key=lambda x: results[x]['test_mse'])
        best_mae = min(results.keys(), key=lambda x: results[x]['test_mae'])
        best_r2 = max(results.keys(), key=lambda x: results[x]['test_r2'])
        fastest = min(results.keys(), key=lambda x: results[x]['total_time'])
        
        f.write("BEST PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best MSE: {best_mse} ({results[best_mse]['test_mse']:.4f})\n")
        f.write(f"Best MAE: {best_mae} ({results[best_mae]['test_mae']:.4f})\n")
        f.write(f"Best R2: {best_r2} ({results[best_r2]['test_r2']:.4f})\n")
        f.write(f"Fastest: {fastest} ({results[fastest]['total_time']:.1f}s)\n")
        
        # Your method performance
        f.write("\nYOUR BRACHISTOCHRONE METHOD:\n")
        f.write("-" * 30 + "\n")
        brach_method = 'Brachistochrone'
        if brach_method in results:
            f.write(f"Test MSE: {results[brach_method]['test_mse']:.4f}\n")
            f.write(f"Test MAE: {results[brach_method]['test_mae']:.4f}\n")
            f.write(f"Test R2: {results[brach_method]['test_r2']:.4f}\n")
            f.write(f"Training Time: {results[brach_method]['total_time']:.1f}s\n")

def main():
    args = get_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    
    # Test dataset
    results = test_electricity_dataset(
        args.sample_size, args.epochs, 
        args.batch_size, device, args.output_dir
    )
    
    # Save numerical summary
    save_numerical_summary(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
