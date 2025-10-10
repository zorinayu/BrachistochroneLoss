# BRACHISTOCHRONE: A Novel Loss Function for Neural Network Training

This repository implements the BRACHISTOCHRONE method, a novel loss function for neural network training that incorporates path optimization principles from physics.

## Overview

The BRACHISTOCHRONE loss function combines traditional task loss with two additional components:
- **L_path**: Path optimization term based on energy minimization
- **L_mono**: Monotonicity constraint term

The method is inspired by the brachistochrone problem in physics, which seeks the path of least time between two points under gravity.

## Installation

```bash
# Clone the repository
git clone https://github.com/zorinayu/Brachistochrone.git
cd Brachistochrone

# Install dependencies
pip install torch torchvision pandas scikit-learn matplotlib seaborn tqdm numpy
```

## Usage

### Basic Usage

```python
from src.brachlearn.losses.brachistochrone import BrachistochroneLoss
from src.brachlearn.models.mlp import MLPClassifier

# Create model
model = MLPClassifier(input_dim=784, hidden_dims=[256, 128], num_classes=10)

# Create loss function
brach_loss = BrachistochroneLoss(beta=1.0, gamma=1.0)

# Training loop
for batch in dataloader:
    x, y = batch
    logits = model(x)
    
    # Calculate BRACHISTOCHRONE loss
    h_list = [x, logits]  # Intermediate representations
    L_path, L_mono = brach_loss(h_list)
    
    # Combine with task loss
    task_loss = F.cross_entropy(logits, y)
    total_loss = task_loss + L_path + L_mono
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

### Testing on Real Datasets

The repository includes comprehensive tests on multiple real-world datasets:

```bash
# Credit Card Fraud Detection
python test_creditcard.py --sample_size 5000 --epochs 10

# NSL-KDD Network Security
python test_nslkdd.py --sample_size 5000 --epochs 10

# Sleep Stage Classification
python test_sleepedfx.py --sample_size 2000 --epochs 10

# MIT-BIH Arrhythmia Detection
python test_mitbih.py --sample_size 2000 --epochs 10

# Combined HAR and Credit Card tests
python test_real_datasets_v2.py --sample_size 5000 --epochs 10
```

## Results

The BRACHISTOCHRONE method has been tested on multiple datasets with the following results:

### Credit Card Fraud Detection
- **Brachistochrone**: 99.90% accuracy, 99.85% F1-score
- **Best baseline**: AdamW (99.90% accuracy, 99.85% F1-score)

### NSL-KDD Network Security
- **Brachistochrone**: 88.50% accuracy, 84.54% F1-score
- **Best baseline**: AdamW (88.60% accuracy, 84.86% F1-score)

### Sleep Stage Classification
- **Brachistochrone**: 71.00% accuracy, 70.66% F1-score
- **Best baseline**: SGD (70.50% accuracy, 70.10% F1-score)

### MIT-BIH Arrhythmia Detection
- **Brachistochrone**: 100.00% accuracy, 100.00% F1-score
- **All methods**: Achieved perfect performance on this dataset

## Method Variants

The repository includes several variants of the BRACHISTOCHRONE method:

1. **Full Brachistochrone**: Includes both L_path and L_mono terms
2. **No L_path**: Removes the path optimization term (β=0)
3. **No L_mono**: Removes the monotonicity term (γ=0)
4. **No L_path+no L_mono**: Pure task loss (β=0, γ=0)

## Architecture

```
src/
├── brachlearn/
│   ├── losses/
│   │   └── brachistochrone.py    # Main loss implementation
│   ├── models/
│   │   └── mlp.py                # MLP classifier
│   └── utils/
│       └── stft.py               # STFT utilities
├── test_*.py                     # Dataset-specific tests
└── outputs/                      # Test results
```

## Key Features

- **GPU Acceleration**: Supports CUDA-enabled PyTorch
- **Multiple Datasets**: Tested on 5+ real-world datasets
- **Comprehensive Evaluation**: Accuracy, F1-score, and training time metrics
- **Modular Design**: Easy to integrate with existing PyTorch workflows
- **Extensive Testing**: 7 different method variants tested

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TQDM

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{brachistochrone2024,
  title={BRACHISTOCHRONE: A Novel Loss Function for Neural Network Training},
  author={Your Name},
  year={2024},
  url={https://github.com/zorinayu/Brachistochrone}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.