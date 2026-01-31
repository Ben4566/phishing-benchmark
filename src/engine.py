import numpy as np
from typing import Dict, Any, Optional
from src.interfaces import IPredictor
from src.benchmark import PerformanceMonitor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class Trainer:
    """
    Lifecycle Orchestrator for Machine Learning experiments.
    
    Responsibilities:
    1. Manages the execution flow: Training -> Inference -> Evaluation.
    2. Decouples the specific model implementation (via IPredictor) from the benchmarking logic.
    3. Integration point for performance monitoring (Time & Resource tracking).
    """
    def __init__(self, model: IPredictor, monitor: PerformanceMonitor):
        self.model = model
        self.monitor = monitor

    def run(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        """
        Executes a complete benchmark cycle.
        
        Args:
            X_train, y_train: Training data.
            X_test, y_test: Hold-out set for final metric calculation.
            X_val, y_val: (Optional) Validation set for early stopping or hyperparameter tuning.
                          If None, the Test set is repurposed for validation.
        """
        # --- Phase 1: Training ---
        self.monitor.start_measurement()
        
        # Validation Strategy: Fallback to Test set if no explicit Validation set is provided
        actual_X_val = X_val if X_val is not None else X_test
        actual_y_val = y_val if y_val is not None else y_test

        self.model.train(X_train, y_train, X_val=actual_X_val, y_val=actual_y_val)
        self.monitor.end_measurement(task_name="Training")

        # --- Phase 2: Inference & Evaluation ---
        self.monitor.start_measurement()
        
        # A) Probability Estimation (Computationally intensive step)
        y_scores = self.model.predict_proba(X_test)
        
        # B) Thresholding (Converting continuous probabilities to binary class labels)
        # We assume a standard decision boundary of 0.5 for binary classification.
        y_pred = (y_scores > 0.5).astype(int) 
        
        # C) Metric Computation
        # Note: Calculation time is included in the inference measurement but is usually negligible compared to model prediction.
        metrics = self._calculate_metrics(y_test, y_pred, y_scores)
        
        self.monitor.end_measurement(task_name="Inference", extra_metrics=metrics)

    def save_model(self, filepath: str):
        """Persists the trained model artifact to disk."""
        self.model.save(filepath)

    def _calculate_metrics(self, y_true, y_pred, y_scores) -> Dict[str, Any]:
        """
        Encapsulates statistical evaluation logic.
        Handles edge cases like division-by-zero or single-class batches.
        """
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        except ValueError:
            # Handles rare cases where confusion matrix shape is unexpected (e.g., only 1 class present)
            fpr = 0.0
        
        # Ensure y_true is a raw numpy array (stripping Pandas metadata if present)
        if hasattr(y_true, "values"): 
            y_true = y_true.values
            
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            # AUC requires more than one class to be defined
            "auc": round(roc_auc_score(y_true, y_scores), 4) if len(np.unique(y_true)) > 1 else 0.0,
            "fpr": round(fpr, 4)
        }