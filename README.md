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
from src.losses.brachistochrone import BrachistochroneLoss
from src.models.mlp import MLPClassifier

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
# Iris Dataset (Classification)
python scripts/test_iris.py --sample_size 150 --epochs 3 --batch_size 32

# Breast Cancer Wisconsin (Classification)
python scripts/test_breast_cancer.py --sample_size 500 --epochs 3 --batch_size 32

# MNIST (Image Classification)
python scripts/test_mnist.py --sample_size 5000 --epochs 3 --batch_size 64

# Fashion-MNIST (Image Classification)
python scripts/test_fashion_mnist.py --sample_size 5000 --epochs 3 --batch_size 64

# AG News (Text Classification)
python scripts/test_agnews.py --sample_size 2000 --epochs 3 --batch_size 64

# Adult Income (Classification)
python scripts/test_adult.py --sample_size 5000 --epochs 10 --batch_size 64

# Wine Quality (Classification)
python scripts/test_wine.py --sample_size 5000 --epochs 10 --batch_size 64

# CIFAR-10 (Image Classification)
python scripts/test_cifar10.py --sample_size 5000 --epochs 10 --batch_size 64

# IMDB Movie Reviews (Text Classification)
python scripts/test_imdb.py --sample_size 5000 --epochs 10 --batch_size 64

# 20 Newsgroups (Text Classification)
python scripts/test_20newsgroups.py --sample_size 2000 --epochs 3 --batch_size 64

# Cora (Paper Classification)
python scripts/test_cora.py --sample_size 2000 --epochs 3 --batch_size 64

# Air Quality (Regression)
python scripts/test_air_quality.py --sample_size 2000 --epochs 3 --batch_size 64

# Electricity Load Diagrams (Regression)
python scripts/test_electricity.py --sample_size 2000 --epochs 3 --batch_size 64

# MovieLens (Rating Prediction)
python scripts/test_movielens.py --sample_size 2000 --epochs 3 --batch_size 64
```

## Results

The BRACHISTOCHRONE method has been tested on multiple datasets with the following results:

### Iris Dataset (Classification)
- **Brachistochrone**: 33.33% accuracy, 16.67% F1-score
- **Best method**: BrachistochroneSGD (66.67% accuracy, 55.56% F1-score)

### Breast Cancer Wisconsin (Classification)
- **Brachistochrone**: 90.00% accuracy, 89.74% F1-score
- **Best method**: BrachistochroneAdam (93.00% accuracy, 92.90% F1-score)

### MNIST (Image Classification)
- **Brachistochrone**: 59.72% accuracy, 51.15% F1-score
- **Best method**: Adam (89.17% accuracy, 88.92% F1-score)

### Fashion-MNIST (Image Classification)
- **Brachistochrone**: 73.90% accuracy, 71.91% F1-score
- **Best method**: BrachistochroneAdam (82.10% accuracy, 81.51% F1-score)

### AG News (Text Classification)
- **Brachistochrone**: 100.00% accuracy, 100.00% F1-score
- **All methods**: Achieved perfect performance on this dataset

### 20 Newsgroups (Text Classification)
- **Brachistochrone**: 32.75% accuracy, 29.32% F1-score
- **Best method**: Brachistochrone (32.75% accuracy, 29.32% F1-score)

### Cora (Paper Classification)
- **Brachistochrone**: 18.75% accuracy, 12.65% F1-score
- **Best method**: Adam (26.00% accuracy, 25.34% F1-score)

### Air Quality (Regression)
- **Brachistochrone**: MSE=0.8012, MAE=0.7077, R²=0.3175
- **Best method**: Adam (MSE=0.0836, MAE=0.2301, R²=0.9288)

### Electricity Load (Regression)
- **Brachistochrone**: MSE=1.0052, MAE=0.8035, R²=0.0938
- **Best method**: Adam (MSE=0.6881, MAE=0.6605, R²=0.3796)

### MovieLens (Rating Prediction)
- **Brachistochrone**: MSE=1.1340, MAE=0.8616, R²=-0.0009
- **Best method**: Brachistochrone (MSE=1.1340, MAE=0.8616, R²=-0.0009)

### Adult Income (Classification)
- **Brachistochrone**: Results available in outputs/adult/
- **Best method**: See detailed results in adult_results.txt

### Wine Quality (Classification)
- **Brachistochrone**: Results available in outputs/wine/
- **Best method**: See detailed results in wine_results.txt

### CIFAR-10 (Image Classification)
- **Brachistochrone**: Results available in outputs/cifar10/
- **Best method**: See detailed results in cifar10_results.txt

### IMDB Movie Reviews (Text Classification)
- **Brachistochrone**: Results available in outputs/imdb/
- **Best method**: See detailed results in imdb_results.txt

## Method Variants

The repository includes several variants of the BRACHISTOCHRONE method:

1. **Full Brachistochrone**: Includes both L_path and L_mono terms
2. **No L_path**: Removes the path optimization term (β=0)
3. **No L_mono**: Removes the monotonicity term (γ=0)
4. **No L_path+no L_mono**: Pure task loss (β=0, γ=0)

## Architecture

```
src/
├── losses/
│   ├── brachistochrone.py        # Main loss implementation
│   └── brachistochrone_pro.py    # Improved loss variants
├── models/
│   └── mlp.py                    # MLP classifier
├── scripts/
│   ├── test_*.py                 # Dataset-specific tests
│   └── generate_figs.py          # Generate result figures
└── outputs/                      # Test results
    ├── iris/
    ├── breast_cancer/
    ├── mnist/
    ├── fashion_mnist/
    ├── agnews/
    ├── 20newsgroups/
    ├── cora/
    ├── air_quality/
    ├── electricity/
    ├── movielens/
    ├── adult/
    ├── wine/
    ├── cifar10/
    └── imdb/
```

## Key Features

- **GPU Acceleration**: Supports CUDA-enabled PyTorch
- **Multiple Datasets**: Tested on 14+ real-world datasets including:
  - Image classification (MNIST, Fashion-MNIST, CIFAR-10)
  - Text classification (AG News, IMDB, 20 Newsgroups)
  - Tabular data (Iris, Breast Cancer, Adult Income, Wine Quality, Cora)
  - Regression tasks (Air Quality, Electricity Load, MovieLens, Boston Housing)
- **Comprehensive Evaluation**: Accuracy, F1-score, and training time metrics
- **Modular Design**: Easy to integrate with existing PyTorch workflows
- **Extensive Testing**: Multiple method variants tested (Brachistochrone, BrachistochroneAdam, BrachistochroneSGD, Adam, SGD)
- **Automatic Result Generation**: Scripts generate detailed results and figures

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