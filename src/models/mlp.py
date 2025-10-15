import torch
from torch import nn

class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for classification tasks
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MLPRegressor(nn.Module):
    """
    Multi-layer perceptron for regression tasks
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=1, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CNN1DClassifier(nn.Module):
    """
    1D CNN for time series classification
    """
    def __init__(self, input_dim, num_classes=2, channels=[64, 128, 256], dropout=0.2):
        super().__init__()
        
        # Reshape input to (B, 1, T) for 1D conv
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        layers = []
        in_channels = 1
        
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes)
        )
    
    def forward(self, x):
        # Reshape for 1D conv: (B, T) -> (B, 1, T)
        x = self.input_proj(x).unsqueeze(1)
        x = self.conv_layers(x)
        return self.classifier(x)
