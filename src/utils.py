import random
import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def seed_everything(seed=42):
    """Setzt alle Random Seeds für Reproduzierbarkeit."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def enforce_imbalance(X, y, target_ratio=1000):
    """
    Simuliert ein Ungleichgewicht in den Daten (z.B. 1:1000).
    Behält alle negativen Samples (Klasse 0) und reduziert positive Samples (Klasse 1).
    """
    y = np.array(y)
    
    # Indizes finden
    benign_indices = np.where(y == 0)[0]
    phishing_indices = np.where(y == 1)[0]
    
    n_benign = len(benign_indices)
    # Berechnen, wie viele Phishing-Samples wir brauchen für das Verhältnis
    n_phishing_needed = int(n_benign / target_ratio)
    
    if n_phishing_needed < 1: 
        n_phishing_needed = 1
        
    # Zufällige Auswahl der Phishing Samples
    if len(phishing_indices) < n_phishing_needed:
        chosen_phishing = phishing_indices
    else:
        chosen_phishing = np.random.choice(phishing_indices, n_phishing_needed, replace=False)
        
    # Zusammenfügen
    keep_indices = np.concatenate([benign_indices, chosen_phishing])
    np.random.shuffle(keep_indices)
    
    y_new = y[keep_indices]
    
    # Unterscheidung Listen (URLs) vs Numpy/Pandas
    if isinstance(X, list):
        X_new = [X[i] for i in keep_indices]
    else:
        # Falls es ein DataFrame ist
        if hasattr(X, 'iloc'):
            X_new = X.iloc[keep_indices]
        else:
            X_new = X[keep_indices]
        
    logger.info(f"--- Imbalance Applied ({target_ratio}:1): {len(benign_indices)} Benign, {len(chosen_phishing)} Phishing ---")
    return X_new, y_new