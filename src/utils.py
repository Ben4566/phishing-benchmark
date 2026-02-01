import random
import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def seed_everything(seed=42):
    """
    Enforces deterministic behavior across all random number generators.
    Crucial for reproducibility in stochastic machine learning processes.
    """
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
    Simulates real-world data scarcity by undersampling the positive class (Phishing).
    
    Args:
        X: Input features (List of URLs or DataFrame).
        y: Labels.
        target_ratio (int): The ratio of Negative:Positive samples (e.g., 1000:1).
    
    Returns:
        tuple: Imbalanced (X_new, y_new) preserving all benign samples.
    """
    y = np.array(y)
    
    # Identify class distribution
    benign_indices = np.where(y == 0)[0]
    phishing_indices = np.where(y == 1)[0]
    
    n_benign = len(benign_indices)
    # Calculate required positive samples to satisfy the ratio
    n_phishing_needed = int(n_benign / target_ratio)
    
    # Boundary check: Ensure at least one positive sample exists
    if n_phishing_needed < 1: 
        n_phishing_needed = 1
        
    # Undersampling Logic: Select random subset of phishing samples
    if len(phishing_indices) < n_phishing_needed:
        chosen_phishing = phishing_indices
    else:
        chosen_phishing = np.random.choice(phishing_indices, n_phishing_needed, replace=False)
        
    # Aggregate and Shuffle
    keep_indices = np.concatenate([benign_indices, chosen_phishing])
    np.random.shuffle(keep_indices)
    
    y_new = y[keep_indices]
    
    # Polymorphic handling for List (raw URLs) vs DataFrame (Feature Vectors)
    if isinstance(X, list):
        X_new = [X[i] for i in keep_indices]
    else:
        # Pandas-compatible slicing
        if hasattr(X, 'iloc'):
            X_new = X.iloc[keep_indices]
        else:
            X_new = X[keep_indices]
        
    logger.info(f"--- Imbalance Constraints Applied ({target_ratio}:1) | Benign: {len(benign_indices)}, Phishing: {len(chosen_phishing)} ---")
    return X_new, y_new