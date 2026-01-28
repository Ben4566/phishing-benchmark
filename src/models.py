import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Any, Union, Optional

# --- 1. CNN Model Architecture ---
class CNNModel(nn.Module):
    """
    1D-CNN architecture optimized for character-level URL analysis.
    Designed to capture local sequential patterns (n-grams) via convolution operations.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super(CNNModel, self).__init__()
        # Embedding layer converts integer-encoded characters into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.global_pool = nn.AdaptiveMaxPool1d(1) # Forces fixed output size regardless of input length
        
        # Fully connected classification head
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5) # Regularization to prevent overfitting
        self.fc2 = nn.Linear(64, 1)
        
        # NOTE: We intentionally omit the Sigmoid activation here.
        # This model outputs raw logits. This allows the use of `BCEWithLogitsLoss`
        # in the training loop, which is numerically more stable than `BCELoss` + `Sigmoid`.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1) # Reshape for Conv1d (Batch, Channels, Seq_Len)
        
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.global_pool(self.relu(self.conv2(x)))
        
        x = x.squeeze(-1) 
        x = self.dropout(self.relu(self.fc1(x)))
        
        return self.fc2(x) # Return Logits

# --- 2. Custom GPU Logistic Regression ---
class GPULogisticRegression(nn.Module):
    """
    Custom implementation of Logistic Regression using PyTorch.
    Standard sklearn LogisticRegression is CPU-bound; this implementation enables
    massive parallelization on GPUs for high-dimensional feature spaces.
    """
    def __init__(self, input_dim: int, device: torch.device):
        super(GPULogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.device = device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Here we use Sigmoid explicitly because we use standard BCELoss in fit()
        return torch.sigmoid(self.linear(x))

    def fit(self, X, y, epochs=100, lr=0.01, batch_size=4096):
        """
        Custom training loop handling mini-batch gradient descent on the GPU.
        """
        self.train()
        # Handle Input: Support both DataFrame/Series and NumPy arrays
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        num_samples = X_tensor.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for _ in range(epochs):
            # Shuffle data each epoch for better convergence
            indices = torch.randperm(num_samples, device=self.device)
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                
                optimizer.zero_grad()
                outputs = self.forward(X_tensor[batch_idx])
                loss = criterion(outputs, y_tensor[batch_idx])
                loss.backward()
                optimizer.step()
                
    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        X_np = X.values if hasattr(X, 'values') else X
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            probs = outputs.cpu().numpy().flatten()
        # Return format matching sklearn: [prob_class_0, prob_class_1]
        return np.vstack(((1 - probs), probs)).T

# --- 3. XGBoost Configuration ---
def get_xgboost_model(use_cuda: bool, n_estimators: int = 100, learning_rate: float = 0.1) -> xgb.XGBClassifier:
    """
    Configures XGBoost with Histogram-based tree method.
    'hist' is significantly faster on large datasets and supports GPU acceleration.
    """
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": 6,                
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    if use_cuda:
        try:
            # Use GPU-accelerated histogram algorithm
            params["tree_method"] = "hist" 
            params["device"] = "cuda"
        except Exception:
            # Fallback to CPU-based histogram if CUDA init fails
            params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist" 

    return xgb.XGBClassifier(**params)

# --- 4. SVM Wrapper ---
def get_svm_model() -> Any:
    """
    Wraps LinearSVC with CalibratedClassifierCV.
    
    Why?
    Standard SVMs output the distance to the hyperplane, not a probability.
    To compute AUC and LogLoss, we need probability estimates.
    CalibratedClassifierCV applies Platt Scaling (Isotonic Regression) to map
    SVM outputs to probabilities.
    """
    base_model = LinearSVC(dual=False) # dual=False is preferred when n_samples > n_features
    return CalibratedClassifierCV(base_model)