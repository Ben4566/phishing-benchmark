import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Any

# --- LOGGING INTEGRATION ---
from src.logger import setup_logger
logger = setup_logger(__name__)

# --- 1. Deep Learning Architecture (CNN) ---
class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network optimized for character-level URL analysis.
    
    Architecture:
    1. Embedding: Maps discrete character IDs to dense vectors.
    2. Feature Extraction: Dual Conv1D layers with MaxPool to capture local sequential patterns.
    3. Aggregation: Global Adaptive Pooling to handle variable-length inputs.
    4. Classification: Dense layers with Dropout for regularization.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Layer 1: Local pattern detection (e.g., "http", "www")
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Layer 2: Higher-level feature composition
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.global_pool = nn.AdaptiveMaxPool1d(1) 
        
        # Classification Head
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1) # Reshape for Conv1D (Batch, Channels, Seq_Len)
        
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.global_pool(self.relu(self.conv2(x)))
        x = x.squeeze(-1) 
        
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x) 

# --- 2. GPU-Accelerated Classical Models ---
class GPULogisticRegression(nn.Module):
    """
    PyTorch implementation of Logistic Regression designed to mimic the Scikit-Learn API.
    
    Purpose:
    Enables massive parallelization (GPU) for simple linear models, which is typically 
    CPU-bound in standard libraries like Sklearn.
    """
    def __init__(self, input_dim: int, device: torch.device, epochs: int = 100, lr: float = 0.01, batch_size: int = 4096):
        super(GPULogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.device = device
        
        # Hyperparameters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    def fit(self, X, y):
        """Standard Sklearn-style training loop adapted for PyTorch tensors."""
        self.train()
        
        # Data Adaption: Handle Pandas Series/DataFrames or Numpy arrays
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        # Memory Transfer
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        num_samples = X_tensor.shape[0]
        batch_size = min(self.batch_size, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Training Loop
        for _ in range(self.epochs):
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
        """Returns probability estimates (N, 2) to maintain Sklearn compatibility."""
        self.eval()
        X_np = X.values if hasattr(X, 'values') else X
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            probs = outputs.cpu().numpy().flatten()
            
        # Stack probabilities for Class 0 and Class 1
        return np.vstack(((1 - probs), probs)).T

    def predict(self, X) -> np.ndarray:
        """Returns hard class labels (0/1). Required for accuracy metrics."""
        scores = self.predict_proba(X)[:, 1]
        return (scores > 0.5).astype(int)

# --- 3. External Library Wrappers ---

def get_xgboost_model(use_cuda: bool, n_estimators: int = 100, learning_rate: float = 0.1) -> xgb.XGBClassifier:
    """Configures XGBoost with optional CUDA acceleration."""
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": 6,                
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    
    if use_cuda:
        try:
            params["tree_method"] = "hist" 
            params["device"] = "cuda"
        except Exception:
            logger.warning("XGBoost CUDA initialization failed. Fallback to CPU histogram.")
            params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist" 

    return xgb.XGBClassifier(**params)

def get_svm_model() -> Any:
    """
    Returns a Linear SVC wrapped in a CalibratedClassifierCV.
    
    Reasoning:
    LinearSVC (SVM) does not output probabilities (predict_proba) by default.
    Calibration (Isotonic/Sigmoid) is required to produce valid confidence scores 
    for AUC/ROC metrics.
    """
    base_model = LinearSVC(dual=False) 
    return CalibratedClassifierCV(base_model)