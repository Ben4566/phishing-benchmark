import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Union

# Interface Contract
from src.interfaces import IPredictor

# --- OPTIONAL DEPENDENCY: SAFETENSORS ---
try:
    from safetensors.torch import save_model as safe_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# --- REPRODUCIBILITY UTILITIES ---
def seed_worker(worker_id):
    """
    Ensures deterministic behavior in PyTorch DataLoaders.
    without this, worker processes might generate identical random sequences.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

class SklearnAdapter(IPredictor):
    """
    Adapter Pattern Implementation for Scikit-Learn compatible models.
    
    Compatible with:
    1. Standard Sklearn models (SVM, RandomForest).
    2. XGBoost (via Sklearn API).
    3. Custom GPULogisticRegression (which implements .fit/.predict).
    """
    def __init__(self, model, use_eval_set: bool = False):
        """
        Args:
            model: The instantiated estimator.
            use_eval_set (bool): If True, passes validation data to .fit() (specifically for XGBoost early stopping).
        """
        self.model = model
        self.use_eval_set = use_eval_set

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Wraps the training logic, injecting validation sets if the underlying model supports it via Duck Typing.
        """
        if self.use_eval_set and X_val is not None and y_val is not None:
            # Duck Typing Check: Does the model support 'eval_set'? (Common in XGBoost/LGBM)
            if hasattr(self.model, "fit"):
                try:
                    # XGBoost often requires explicit float types, avoiding object/int ambiguity
                    self.model.fit(
                        X_train.astype(float), 
                        y_train.astype(float),
                        eval_set=[(X_val.astype(float), y_val.astype(float))],
                        verbose=False
                    )
                except TypeError:
                    # Fallback: Model has .fit() but arguments differ (e.g. Standard Sklearn)
                    self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        if hasattr(X, "astype"):
            X = X.astype(float)
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Standardizes probability output to a 1D array for the positive class.
        """
        if hasattr(X, "astype"):
            X = X.astype(float)
            
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            
            # Scikit-Learn returns (N, 2) -> [Prob_Class0, Prob_Class1]
            if probs.shape[1] == 2:
                return probs[:, 1]
            
            # Fallback for some custom implementations returning (N, 1)
            return probs[:, 0]
        else:
            raise NotImplementedError("The underlying model does not support probability estimation.")

    def save(self, path: str):
        """Persists the model using Joblib (standard for Sklearn/XGBoost)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)


class PyTorchAdapter(IPredictor):
    """
    Adapter Pattern Implementation for Deep Learning models (PyTorch).
    
    Responsibilities:
    1. Manages DataLoaders.
    2. Encapsulates the Training Loop.
    3. Handles Device Management (CPU/GPU).
    4. Implements Mixed Precision (AMP) for performance.
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: nn.Module, 
                 device: torch.device,
                 batch_size: int = 128,
                 epochs: int = 10,
                 seed: int = 42):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        
        # Automatic Mixed Precision (AMP) Scaler initialization
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    def _create_dataloader(self, X, y, shuffle=False):
        """Internal factory for creating reproducible DataLoaders."""
        # Type Coercion: Ensure inputs are Tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.long) # Long required for Embedding Layers
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            
        dataset = TensorDataset(X, y)
        
        # Generator ensures reproducibility across worker threads
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(self.device.type == 'cuda'),
            worker_init_fn=seed_worker,
            generator=g
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.train()
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                # AMP Context: Casts operations to float16 where safe
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        
        # Create dummy labels to reuse the _create_dataloader logic strictly for batching
        dummy_y = np.zeros(len(X))
        loader = self._create_dataloader(X, dummy_y, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    logits = self.model(X_batch)
                
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.float().cpu().numpy())
        
        # Return 1D array
        return np.array(all_probs).flatten()

    def predict(self, X) -> np.ndarray:
        scores = self.predict_proba(X)
        return (scores > 0.5).astype(int)

    def save(self, path: str):
        """
        Saves the model state. Prefers Safetensors format if available (safer/faster serialization).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if SAFETENSORS_AVAILABLE:
            if not path.endswith(".safetensors"):
                path += ".safetensors"
            try:
                # Safetensors is strict about device placement; ensuring compatibility
                safe_save(self.model, path)
            except Exception:
                # Fallback to standard PyTorch pickle if Safetensors constraints are unmet
                torch.save(self.model.state_dict(), path.replace(".safetensors", ".pt"))
        else:
            torch.save(self.model.state_dict(), path + ".pt")