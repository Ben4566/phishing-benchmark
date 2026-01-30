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

# --- 1. CNN Model Architecture ---
class CNNModel(nn.Module):
    # (Bleibt unverändert)
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.global_pool = nn.AdaptiveMaxPool1d(1) 
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1) 
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.global_pool(self.relu(self.conv2(x)))
        x = x.squeeze(-1) 
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x) 

# --- 2. Custom GPU Logistic Regression ---
class GPULogisticRegression(nn.Module):
    """
    Verhält sich wie ein Sklearn-Modell, läuft aber auf GPU (PyTorch).
    """
    def __init__(self, input_dim: int, device: torch.device, epochs: int = 100, lr: float = 0.01, batch_size: int = 4096):
        super(GPULogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.device = device
        
        # Hyperparameter speichern
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    def fit(self, X, y):
        self.train()
        
        # Konvertierung
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        num_samples = X_tensor.shape[0]
        batch_size = min(self.batch_size, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))
        
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
        self.eval()
        X_np = X.values if hasattr(X, 'values') else X
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            probs = outputs.cpu().numpy().flatten()
        # Sklearn erwartet (N, 2) Output für binary classification
        return np.vstack(((1 - probs), probs)).T

    # --- DIESE METHODE HAT GEFEHLT ---
    def predict(self, X) -> np.ndarray:
        """Gibt harte Labels (0/1) zurück, wie Sklearn es erwartet."""
        scores = self.predict_proba(X)[:, 1] # Wahrscheinlichkeit für Klasse 1
        return (scores > 0.5).astype(int)
    # ---------------------------------

# --- 3. XGBoost & SVM ---
# (Bleiben unverändert, sind okay)
def get_xgboost_model(use_cuda: bool, n_estimators: int = 100, learning_rate: float = 0.1) -> xgb.XGBClassifier:
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
            logger.warning("XGBoost CUDA init failed. Falling back to CPU histogram.")
            params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist" 

    return xgb.XGBClassifier(**params)

def get_svm_model() -> Any:
    base_model = LinearSVC(dual=False) 
    return CalibratedClassifierCV(base_model)