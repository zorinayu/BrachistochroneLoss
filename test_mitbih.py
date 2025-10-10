#!/usr/bin/env python3
"""
Test BRACHISTOCHRONE method on MIT-BIH Arrhythmia Database dataset
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
    parser = argparse.ArgumentParser(description='Test BRACHISTOCHRONE on MIT-BIH dataset')
    parser.add_argument('--sample_size', type=int, default=5000, help='Sample size for testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/mitbih', help='Output directory')
    return parser.parse_args()

class MITBIHDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data/MIT-BIH Arrhythmia Database', 
                 sample_size=None, test_size=0.2, random_state=42):
        
        # Load ECG data and annotations
        all_data = []
        all_labels = []
        
        # Get all ECG files
        ecg_files = [f for f in os.listdir(data_dir) if f.endswith('_ekg.csv')]
        
        print(f"Found {len(ecg_files)} ECG files")
        
        for ecg_file in ecg_files[:10]:  # Limit to first 10 files for testing
            patient_id = ecg_file.split('_')[0]
            annotation_file = f"{patient_id}_annotations_1.csv"
            
            ecg_path = os.path.join(data_dir, ecg_file)
            ann_path = os.path.join(data_dir, annotation_file)
            
            if os.path.exists(ecg_path) and os.path.exists(ann_path):
                print(f"Processing {patient_id}...")
                
                # Load ECG data (load more data)
                ecg_df = pd.read_csv(ecg_path, nrows=sample_size//len(ecg_files) if sample_size else 5000)
                
                # Handle different column names across files
                if 'MLII' in ecg_df.columns:
                    ecg_data = ecg_df['MLII'].values
                elif 'V5' in ecg_df.columns:
                    ecg_data = ecg_df['V5'].values
                elif 'V1' in ecg_df.columns:
                    ecg_data = ecg_df['V1'].values
                elif 'V2' in ecg_df.columns:
                    ecg_data = ecg_df['V2'].values
                else:
                    # Use the first numeric column
                    numeric_cols = ecg_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        ecg_data = ecg_df[numeric_cols[0]].values
                    else:
                        continue
                
                print(f"  ECG data length: {len(ecg_data)}")
                
                # Load annotations
                ann_df = pd.read_csv(ann_path)
                
                # Create segments around annotations
                window_size = 32  # Smaller window size for shorter data
                for _, ann in ann_df.iterrows():
                    idx = ann['index']
                    label = ann['annotation_symbol']
                    
                    # Skip if annotation is outside data range
                    if idx < window_size//2 or idx >= len(ecg_data) - window_size//2:
                        continue
                    
                    # Extract segment
                    start = idx - window_size//2
                    end = idx + window_size//2
                    segment = ecg_data[start:end]
                    
                    if len(segment) == window_size:
                        all_data.append(segment)
                        all_labels.append(label)
                
                # If no annotations found, create random segments
                if len(all_data) == 0:
                    print(f"  No annotations found for {patient_id}, creating random segments...")
                    n_segments = min(50, len(ecg_data) // window_size)
                    for i in range(n_segments):
                        start = i * window_size
                        end = start + window_size
                        if end <= len(ecg_data):
                            segment = ecg_data[start:end]
                            all_data.append(segment)
                            all_labels.append('N')  # Default to normal
                
                print(f"  Loaded {len(all_data)} segments from {patient_id}")
        
        if not all_data:
            raise ValueError("No data loaded from MIT-BIH Database")
        
        # Convert to numpy arrays
        X = np.array(all_data)
        y = np.array(all_labels)
        
        # Encode labels (simplify to normal vs abnormal)
        normal_labels = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'E']
        y_binary = np.array([1 if label in normal_labels else 0 for label in y])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
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
        self.n_classes = 2
        
        print(f"MIT-BIH Dataset: {len(self.X_train)} train, {len(self.X_test)} test")
        print(f"Features: {self.n_features}, Classes: {self.n_classes}")
        print(f"Normal rate: {y_binary.mean():.4f}")

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

def test_mitbih_dataset(sample_size, epochs, batch_size, device, output_dir):
    """Test all methods on MIT-BIH dataset"""
    print("=" * 60)
    print("TESTING MIT-BIH ARRHYTHMIA DATASET")
    print("=" * 60)
    
    # Load dataset
    dataset = MITBIHDataset(sample_size=sample_size)
    
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
    
    with open(os.path.join(output_dir, 'mitbih_results.txt'), 'w') as f:
        f.write("BRACHISTOCHRONE METHOD TEST RESULTS - MIT-BIH ARRHYTHMIA DATASET\n")
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
    results = test_mitbih_dataset(
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
