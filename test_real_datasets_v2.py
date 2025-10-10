#!/usr/bin/env python3
"""
Test BRACHISTOCHRONE method and its variants on real datasets using GPU acceleration
Using the same 7 baselines from gpu_benchmark.py
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
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Import our modules
from src.brachlearn.data.real_datasets import get_dataset
from src.brachlearn.models.mlp import MLPClassifier, CNN1DClassifier
from src.brachlearn.losses.brachistochrone import BrachistochroneLoss
# from src.brachlearn.utils.metrics import snr_db

def get_args():
    parser = argparse.ArgumentParser(description='Test BRACHISTOCHRONE on real datasets')
    parser.add_argument('--dataset', type=str, default='har', 
                       choices=['har', 'creditcard'], help='Dataset to test')
    parser.add_argument('--sample_size', type=int, default=3000, help='Sample size for large datasets')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='outputs/real_dataset_tests_v2', help='Output directory')
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device with GPU info"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device_arg)
    
    return device

def get_methods():
    """Get the same 7 methods from gpu_benchmark.py"""
    return {
        'Brachistochrone': {
            'optimizer': lambda model: AdamW(model.parameters(), lr=0.001),
            'brach_loss': BrachistochroneLoss(alpha=1.0, beta=0.1, gamma=0.1),
            'color': 'red'
        },
        'Brachistochrone (no L_path)': {
            'optimizer': lambda model: AdamW(model.parameters(), lr=0.001),
            'brach_loss': BrachistochroneLoss(alpha=1.0, beta=0.0, gamma=0.1),
            'color': 'orange'
        },
        'Brachistochrone (no L_mono)': {
            'optimizer': lambda model: AdamW(model.parameters(), lr=0.001),
            'brach_loss': BrachistochroneLoss(alpha=1.0, beta=0.1, gamma=0.0),
            'color': 'purple'
        },
        'Brachistochrone (no L_path+no L_mono)': {
            'optimizer': lambda model: AdamW(model.parameters(), lr=0.001),
            'brach_loss': BrachistochroneLoss(alpha=1.0, beta=0.0, gamma=0.0),
            'color': 'pink'
        },
        'AdamW': {
            'optimizer': lambda model: AdamW(model.parameters(), lr=0.001),
            'brach_loss': None,
            'color': 'blue'
        },
        'Adam': {
            'optimizer': lambda model: Adam(model.parameters(), lr=0.001),
            'brach_loss': None,
            'color': 'green'
        },
        'SGD': {
            'optimizer': lambda model: SGD(model.parameters(), lr=0.001, momentum=0.9),
            'brach_loss': None,
            'color': 'brown'
        }
    }

def train_model(model, train_loader, val_loader, method_config, device, epochs, num_classes):
    """Train a single model"""
    optimizer = method_config['optimizer'](model)
    criterion = nn.CrossEntropyLoss()
    brach_loss = method_config['brach_loss']
    
    train_losses = []
    val_accuracies = []
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if brach_loss is not None:
                # BRACHISTOCHRONE method
                # Create intermediate representations by reshaping to 1D sequences
                batch_size = x.shape[0]
                
                # Reshape features to 1D sequences for STFT
                x_1d = x.view(batch_size, -1)  # (B, features)
                logits = model(x)
                logits_1d = logits.view(batch_size, -1)  # (B, num_classes)
                
                # Pad sequences to make them suitable for STFT
                min_len = 256
                max_len = max(x_1d.shape[1], logits_1d.shape[1], min_len)
                
                if x_1d.shape[1] < max_len:
                    x_1d = torch.nn.functional.pad(x_1d, (0, max_len - x_1d.shape[1]))
                if logits_1d.shape[1] < max_len:
                    logits_1d = torch.nn.functional.pad(logits_1d, (0, max_len - logits_1d.shape[1]))
                
                # Create intermediate representations
                h_list = [x_1d, logits_1d]
                
                # Calculate losses
                L_task = criterion(logits, y)
                L_path, L_mono = brach_loss(h_list)
                total_loss = L_task + L_path + L_mono
            else:
                # Traditional method
                logits = model(x)
                total_loss = criterion(logits, y)
                L_path = torch.tensor(0.0)
                L_mono = torch.tensor(0.0)
            
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(total_loss.item())
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        train_losses.append(np.mean(epoch_losses))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        print(f"  Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Acc={val_acc:.4f}, Time={epoch_time:.1f}s")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'final_accuracy': val_accuracies[-1],
        'final_loss': train_losses[-1],
        'total_time': sum(epoch_times),
        'avg_epoch_time': np.mean(epoch_times)
    }

def test_dataset(dataset_name, sample_size, epochs, batch_size, device, output_dir):
    """Test all methods on a dataset"""
    print(f"\n{'='*60}")
    print(f"TESTING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Load dataset
    train_dataset, val_dataset, test_dataset, input_dim, num_classes = get_dataset(
        dataset_name, sample_size=sample_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset: {dataset_name}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Input dim: {input_dim}")
    print(f"Num classes: {num_classes}")
    
    methods = get_methods()
    results = {}
    
    for method_name, method_config in methods.items():
        print(f"\nTraining {method_name}...")
        
        # Create model
        model = MLPClassifier(input_dim, hidden_dims=[256, 128, 64], num_classes=num_classes).to(device)
        
        # Train
        train_results = train_model(model, train_loader, val_loader, method_config, device, epochs, num_classes)
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.numel()
                test_predictions.extend(pred.cpu().numpy())
                test_targets.extend(y.cpu().numpy())
        
        test_accuracy = test_correct / test_total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_targets, test_predictions, average='weighted'
        )
        
        try:
            auc = roc_auc_score(test_targets, test_predictions, multi_class='ovr')
        except:
            auc = 0.0
        
        results[method_name] = {
            'test_accuracy': test_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_auc': auc,
            'final_accuracy': train_results['final_accuracy'],
            'final_loss': train_results['final_loss'],
            'total_time': train_results['total_time'],
            'avg_epoch_time': train_results['avg_epoch_time'],
            'train_losses': train_results['train_losses'],
            'val_accuracies': train_results['val_accuracies'],
            'epoch_times': train_results['epoch_times']
        }
        
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1: {f1:.4f}")
        print(f"  Total Time: {train_results['total_time']:.1f}s")
    
    return results

def save_numerical_summary(results, dataset_name, output_dir):
    """Save only numerical summary"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical summary
    with open(os.path.join(output_dir, 'numerical_results_summary.txt'), 'w') as f:
        f.write("BRACHISTOCHRONE METHOD TEST RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"DATASET: {dataset_name.upper()}\n")
        f.write("-" * 30 + "\n")
        
        f.write(f"{'Method':<35} {'Test Acc':<10} {'Test F1':<10} {'Time(s)':<10}\n")
        f.write("-" * 75 + "\n")
        
        for method, result in results.items():
            f.write(f"{method:<35} {result['test_accuracy']:<10.4f} "
                   f"{result['test_f1']:<10.4f} {result['total_time']:<10.1f}\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
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
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test dataset
    results = test_dataset(
        args.dataset, args.sample_size, args.epochs, 
        args.batch_size, device, args.output_dir
    )
    
    # Save only numerical summary
    save_numerical_summary(results, args.dataset, args.output_dir)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary
    print(f"\n{args.dataset.upper()} DATASET SUMMARY:")
    print("-" * 30)
    
    best_accuracy = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    brach_method = 'Brachistochrone'
    
    print(f"Best Accuracy: {best_accuracy} ({results[best_accuracy]['test_accuracy']:.4f})")
    if brach_method in results:
        print(f"Your Method: {brach_method} ({results[brach_method]['test_accuracy']:.4f})")
        print(f"Improvement: {results[brach_method]['test_accuracy'] - results[best_accuracy]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
