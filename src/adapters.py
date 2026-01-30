import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Union

# Importiere das Interface
from src.interfaces import IPredictor

try:
    from safetensors.torch import save_model as safe_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Hilfsfunktion für Determinismus im Dataloader (aus train.py übernommen)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

class SklearnAdapter(IPredictor):
    """
    Adapter für Scikit-Learn kompatible Modelle (SVM, XGBoost) 
    und Ihre Custom GPULogisticRegression (da sie .fit/.predict implementiert).
    """
    def __init__(self, model, use_eval_set: bool = False):
        """
        :param model: Das instanziierte Modell (z.B. XGBClassifier oder LinearSVC)
        :param use_eval_set: Wenn True, wird X_val beim Training an XGBoost übergeben.
        """
        self.model = model
        self.use_eval_set = use_eval_set

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # OCP/LSP: Spezifische Logik für Modelle mit Early Stopping (z.B. XGBoost)
        # wird hier gekapselt. Der Aufrufer merkt davon nichts.
        if self.use_eval_set and X_val is not None and y_val is not None:
            # Check if model supports eval_set (Duck Typing)
            if hasattr(self.model, "fit"):
                # XGBoost erwartet float types oft explizit
                try:
                    self.model.fit(
                        X_train.astype(float), 
                        y_train.astype(float),
                        eval_set=[(X_val.astype(float), y_val.astype(float))],
                        verbose=False
                    )
                except TypeError:
                    # Fallback falls Parameter nicht unterstützt werden
                    self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        # Sicherstellen, dass Input korrektes Format hat
        if hasattr(X, "astype"):
            X = X.astype(float)
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if hasattr(X, "astype"):
            X = X.astype(float)
            
        # Sklearn gibt (N, 2) zurück -> [prob_0, prob_1]
        # Wir wollen nur prob_1 für Konsistenz mit CNN
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.shape[1] == 2:
                return probs[:, 1]
            return probs[:, 0] # Fallback
        else:
            raise NotImplementedError("Modell unterstützt kein predict_proba")

    def save(self, path: str):
        # Sklearn Modelle speichern wir klassisch mit joblib
        # Pfad-Ordner sicherstellen
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)


class PyTorchAdapter(IPredictor):
    """
    Adapter für PyTorch Modelle (z.B. CNN).
    Kapselt Dataloader-Erstellung, Training-Loop, GPU-Handling und AMP.
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
        
        # AMP Scaler für Performance (aus train.py übernommen)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    def _create_dataloader(self, X, y, shuffle=False):
        # Konvertierung zu Tensoren, falls noch nicht geschehen
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.long) # Long für Embeddings (CNN)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            
        dataset = TensorDataset(X, y)
        
        # Generator für Reproduzierbarkeit
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

                # Mixed Precision Training
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        
        # Dummy Labels erstellen, da DataLoader (X, y) erwartet
        # Dies vereinfacht die Wiederverwendung von _create_dataloader
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
        
        # Flattened Array zurückgeben
        return np.array(all_probs).flatten()

    def predict(self, X) -> np.ndarray:
        scores = self.predict_proba(X)
        return (scores > 0.5).astype(int)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if SAFETENSORS_AVAILABLE:
            if not path.endswith(".safetensors"):
                path += ".safetensors"
            # Safetensors erwartet, dass das Modell auf CPU ist oder direkt gehandhabt wird
            # Wir speichern das State Dict
            try:
                safe_save(self.model, path)
            except Exception:
                # Fallback falls Safetensors strictness fehlschlägt
                torch.save(self.model.state_dict(), path.replace(".safetensors", ".pt"))
        else:
            torch.save(self.model.state_dict(), path + ".pt")