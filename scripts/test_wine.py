#!/usr/bin/env python3
"""
Test BRACHISTOCHRONE method on Wine Quality dataset
"""

import os
import json
import time
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, Adam, SGD
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import zipfile

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.mlp import MLPClassifier
from src.losses.brachistochrone import BrachistochroneLoss
from src.losses.brachistochrone_pro import BrachistochroneAdam, BrachistochroneSGD

def get_args():
    parser = argparse.ArgumentParser(description='Test BRACHISTOCHRONE on Wine Quality dataset')
    parser.add_argument('--sample_size', type=int, default=5000, help='Sample size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/wine', help='Output directory')
    return parser.parse_args()

class WineDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data/winequality-red.csv', sample_size=None, test_size=0.2, random_state=42):
        
        print("Loading Wine Quality dataset...")
        
        # Check if the CSV file exists directly
        if not os.path.exists(data_path):
            # Try alternative paths
            alt_paths = ['data/winequality-red.csv', '../data/winequality-red.csv']
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Wine data file not found. Tried: {data_path}")
        
        print(f"Loading wine data from: {data_path}")
        
        df = pd.read_csv(data_path, sep=';')
        
        # Check if this is red or white wine data
        if 'quality' in df.columns:
            # This is the wine quality dataset
            X = df.drop('quality', axis=1)
            y = df['quality']
        else:
            # Try to find quality column
            quality_cols = [col for col in df.columns if 'quality' in col.lower()]
            if quality_cols:
                X = df.drop(quality_cols[0], axis=1)
                y = df[quality_cols[0]]
            else:
                raise ValueError("No quality column found in wine dataset")
        
        # Convert quality to categorical (bin into good/bad)
        # Quality scores 3-6 are "bad", 7-9 are "good"
        y_binary = (y >= 7).astype(int)
        
        # Sample data if specified
        if sample_size and sample_size < len(df):
            indices = np.random.choice(len(df), sample_size, replace=False)
            X = X.iloc[indices]
            y_binary = y_binary.iloc[indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.long)
        
        self.n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y_train))
        
        print(f"Wine Dataset: {len(self.X_train)} train, {len(self.X_test)} test")
        print(f"Features: {self.n_features}, Classes: {self.n_classes}")
        print(f"Quality distribution: {np.bincount(y_train)}")

def train_model(model, train_loader, val_loader, epochs, device, brach_loss=None, optimizer_type='adamw'):
    """Train model with or without BRACHISTOCHRONE loss"""
    if optimizer_type == 'mtng':
        optimizer = MTNG(model.parameters(), lr=0.001)
    elif optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=0.001)
    elif optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=0.001)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=0.001)
    else:
        optimizer = AdamW(model.parameters(), lr=0.001)
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        if optimizer_type == 'mtng':
            # MTNG optimizer - simplified approach
            optimizer.zero_grad()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                task_loss = criterion(logits, y)
                
                # Calculate BRACHISTOCHRONE loss
                if brach_loss is not None and brach_loss != 'mtng':
                    batch_size = x.shape[0]
                    x_flat = x.view(batch_size, -1)
                    logits_flat = logits.view(batch_size, -1)
                    
                    h_list = [x_flat, logits_flat]
                    L_path, L_mono = brach_loss(h_list)
                    total_loss += task_loss + L_path + L_mono
                else:
                    total_loss += task_loss
            
            total_loss.backward()
            optimizer.step()
            epoch_loss = total_loss.item()
        else:
            # Standard optimizers
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits = model(x)
                task_loss = criterion(logits, y)
                
                # Calculate BRACHISTOCHRONE loss
                if brach_loss is not None:
                    # For tabular data, use flattened features
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
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Acc={val_acc:.4f}")
    
    total_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'total_time': total_time,
        'final_accuracy': val_accuracies[-1]
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
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return accuracy, f1

def test_wine_dataset(sample_size, epochs, batch_size, device, output_dir):
    """Test all methods on Wine Quality dataset"""
    print("=" * 60)
    print("TESTING WINE QUALITY DATASET")
    print("=" * 60)
    
    # Load dataset
    dataset = WineDataset(sample_size=sample_size)
    
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
                model = MLPClassifier(dataset.n_features, [64, 32], dataset.n_classes).to(device)
                
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
                test_accuracy, test_f1 = evaluate_model(model, test_loader, device)
                
                method_key = f"{method_name}_{epoch_count}epochs"
                results[method_key] = {
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'total_time': train_results['total_time'],
                    'final_accuracy': train_results['final_accuracy'],
                    'epochs': epoch_count
                }
        else:
            # Single epoch for Brachistochrone methods
            # Create model
            model = MLPClassifier(dataset.n_features, [64, 32], dataset.n_classes).to(device)
            
            # Determine optimizer type
            optimizer_type = 'adamw'
            
            # Train model
            train_results = train_model(model, train_loader, val_loader, method_epochs, device, brach_loss, optimizer_type)
            
            # Evaluate on test set
            test_accuracy, test_f1 = evaluate_model(model, test_loader, device)
            
            results[method_name] = {
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'total_time': train_results['total_time'],
                'final_accuracy': train_results['final_accuracy'],
                'epochs': method_epochs
            }
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  Total Time: {train_results['total_time']:.1f}s ({method_epochs} epoch)")
    
    return results

def save_numerical_summary(results, output_dir):
    """Save numerical summary"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'wine_results.txt'), 'w') as f:
        f.write("BRACHISTOCHRONE METHOD TEST RESULTS - WINE QUALITY DATASET\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"{'Method':<35} {'Test Acc':<10} {'Test F1':<10} {'Time(s)':<15} {'Epochs':<10}\n")
        f.write("-" * 85 + "\n")
        
        for method, result in results.items():
            epochs_info = f"{result['epochs']} epoch{'s' if result['epochs'] > 1 else ''}"
            f.write(f"{method:<35} {result['test_accuracy']:<10.4f} "
                   f"{result['test_f1']:<10.4f} {result['total_time']:<15.1f} {epochs_info:<10}\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        # Find best methods
        best_accuracy = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_f1 = max(results.keys(), key=lambda x: results[x]['test_f1'])
        fastest = min(results.keys(), key=lambda x: results[x]['total_time'])
        
        f.write("BEST PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best Accuracy: {best_accuracy} ({results[best_accuracy]['test_accuracy']:.4f})\n")
        f.write(f"Best F1: {best_f1} ({results[best_f1]['test_f1']:.4f})\n")
        f.write(f"Fastest: {fastest} ({results[fastest]['total_time']:.1f}s)\n")
        
        # Your method performance
        f.write("\nYOUR BRACHISTOCHRONE METHOD:\n")
        f.write("-" * 30 + "\n")
        brach_method = 'Brachistochrone'
        if brach_method in results:
            f.write(f"Test Accuracy: {results[brach_method]['test_accuracy']:.4f}\n")
            f.write(f"Test F1: {results[brach_method]['test_f1']:.4f}\n")
            f.write(f"Training Time: {results[brach_method]['total_time']:.1f}s\n")

def main():
    args = get_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    
    # Test dataset
    results = test_wine_dataset(
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
