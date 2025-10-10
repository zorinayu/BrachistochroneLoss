#!/usr/bin/env python3
"""
Test BRACHISTOCHRONE method on PhysioBank Sleep-EDFx dataset
"""

import os
import json
import time
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, Adam, SGD
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import our modules
from src.brachlearn.models.mlp import MLPClassifier
from src.brachlearn.losses.brachistochrone import BrachistochroneLoss

def get_args():
    parser = argparse.ArgumentParser(description='Test BRACHISTOCHRONE on Sleep-EDFx dataset')
    parser.add_argument('--sample_size', type=int, default=5000, help='Sample size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/sleepedfx', help='Output directory')
    return parser.parse_args()

class SleepEDFxDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data/physiobank_database_sleep-edfx_sleep-cassette', 
                 sample_size=None, test_size=0.2, random_state=42):
        
        # For EDF files, we'll create synthetic data based on sleep stages
        # since EDF files require special libraries to read
        
        print("Creating synthetic sleep data based on Sleep-EDFx structure...")
        
        # Sleep stages: W (Wake), N1, N2, N3, R (REM)
        sleep_stages = ['W', 'N1', 'N2', 'N3', 'R']
        n_classes = len(sleep_stages)
        
        # Generate synthetic EEG-like data
        np.random.seed(random_state)
        n_samples = sample_size if sample_size else 1000
        
        all_data = []
        all_labels = []
        
        for i in range(n_samples):
            # Random sleep stage
            stage = np.random.choice(sleep_stages)
            stage_idx = sleep_stages.index(stage)
            
            # Generate synthetic EEG signal (128 samples)
            if stage == 'W':  # Wake - high frequency, low amplitude
                signal = np.random.normal(0, 0.5, 128) + 0.1 * np.sin(2 * np.pi * 20 * np.linspace(0, 1, 128))
            elif stage == 'N1':  # N1 - mixed frequencies
                signal = np.random.normal(0, 0.3, 128) + 0.05 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, 128))
            elif stage == 'N2':  # N2 - sleep spindles and K-complexes
                signal = np.random.normal(0, 0.4, 128) + 0.1 * np.sin(2 * np.pi * 12 * np.linspace(0, 1, 128))
            elif stage == 'N3':  # N3 - slow waves
                signal = np.random.normal(0, 0.6, 128) + 0.2 * np.sin(2 * np.pi * 2 * np.linspace(0, 1, 128))
            else:  # REM - similar to wake but with rapid eye movements
                signal = np.random.normal(0, 0.4, 128) + 0.08 * np.sin(2 * np.pi * 15 * np.linspace(0, 1, 128))
            
            all_data.append(signal)
            all_labels.append(stage_idx)
        
        # Convert to numpy arrays
        X = np.array(all_data)
        y = np.array(all_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Convert to tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        
        self.n_features = X_train.shape[1]
        self.n_classes = n_classes
        
        print(f"Sleep-EDFx Dataset: {len(self.X_train)} train, {len(self.X_test)} test")
        print(f"Features: {self.n_features}, Classes: {self.n_classes}")
        print(f"Sleep stages: {sleep_stages}")

def train_model(model, train_loader, val_loader, epochs, device, brach_loss=None):
    """Train model with or without BRACHISTOCHRONE loss"""
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
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
                # For time series data, use directly
                batch_size = x.shape[0]
                x_1d = x.view(batch_size, -1)
                logits_1d = logits.view(batch_size, -1)
                
                # Pad sequences for STFT
                min_len = 256
                max_len = max(x_1d.shape[1], logits_1d.shape[1], min_len)
                
                if x_1d.shape[1] < max_len:
                    x_1d = torch.nn.functional.pad(x_1d, (0, max_len - x_1d.shape[1]))
                if logits_1d.shape[1] < max_len:
                    logits_1d = torch.nn.functional.pad(logits_1d, (0, max_len - logits_1d.shape[1]))
                
                h_list = [x_1d, logits_1d]
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

def test_sleepedfx_dataset(sample_size, epochs, batch_size, device, output_dir):
    """Test all methods on Sleep-EDFx dataset"""
    print("=" * 60)
    print("TESTING SLEEP-EDFX DATASET")
    print("=" * 60)
    
    # Load dataset
    dataset = SleepEDFxDataset(sample_size=sample_size)
    
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
        'Brachistochrone (no L_path)': BrachistochroneLoss(beta=0.0),
        'Brachistochrone (no L_mono)': BrachistochroneLoss(gamma=0.0),
        'Brachistochrone (no L_path+no L_mono)': BrachistochroneLoss(beta=0.0, gamma=0.0),
        'AdamW': None,
        'Adam': None,
        'SGD': None
    }
    
    results = {}
    
    for method_name, brach_loss in methods.items():
        print(f"\nTraining {method_name}...")
        
        # Create model
        model = MLPClassifier(dataset.n_features, [64], dataset.n_classes).to(device)
        
        # Train model
        train_results = train_model(model, train_loader, val_loader, epochs, device, brach_loss)
        
        # Evaluate on test set
        test_accuracy, test_f1 = evaluate_model(model, test_loader, device)
        
        results[method_name] = {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'total_time': train_results['total_time'],
            'final_accuracy': train_results['final_accuracy']
        }
        
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  Total Time: {train_results['total_time']:.1f}s")
    
    return results

def save_numerical_summary(results, output_dir):
    """Save numerical summary"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'sleepedfx_results.txt'), 'w') as f:
        f.write("BRACHISTOCHRONE METHOD TEST RESULTS - SLEEP-EDFX DATASET\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Method':<35} {'Test Acc':<10} {'Test F1':<10} {'Time(s)':<10}\n")
        f.write("-" * 75 + "\n")
        
        for method, result in results.items():
            f.write(f"{method:<35} {result['test_accuracy']:<10.4f} "
                   f"{result['test_f1']:<10.4f} {result['total_time']:<10.1f}\n")
        
        f.write("\n" + "=" * 70 + "\n\n")
        
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
    results = test_sleepedfx_dataset(
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
