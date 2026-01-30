import numpy as np
from typing import Dict, Any
from src.interfaces import IPredictor
from src.benchmark import PerformanceMonitor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class Trainer:
    """
    Steuert den Trainings- und Evaluationsprozess.
    Kennt keine Details über das Modell, nur das Interface IPredictor.
    """
    def __init__(self, model: IPredictor, monitor: PerformanceMonitor):
        self.model = model
        self.monitor = monitor

    def run(self, X_train, y_train, X_test, y_test):
        """
        Führt den kompletten Benchmark-Zyklus aus: Training -> Inference -> Metriken.
        """
        # 1. Training
        self.monitor.start_measurement()
        # X_test/y_test wird hier als Validierungsset übergeben (für Early Stopping bei XGBoost)
        self.model.train(X_train, y_train, X_val=X_test, y_val=y_test)
        self.monitor.end_measurement(task_name="Training")

        # 2. Inference
        self.monitor.start_measurement()
        # Wir holen Wahrscheinlichkeiten (für AUC) und harte Labels
        y_scores = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
        
        # 3. Metriken berechnen
        metrics = self._calculate_metrics(y_test, y_pred, y_scores)
        
        # Speichern & Loggen
        self.monitor.end_measurement(task_name="Inference", extra_metrics=metrics)
        
    def save_model(self, filepath: str):
        self.model.save(filepath)

    def _calculate_metrics(self, y_true, y_pred, y_scores) -> Dict[str, Any]:
        """Kapselt die Metrik-Logik (vorher in train.py)."""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        except ValueError:
            fpr = 0.0
        
        # Um sicherzugehen, dass y_true numeric ist
        if hasattr(y_true, "values"): 
            y_true = y_true.values
            
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_true, y_scores), 4) if len(np.unique(y_true)) > 1 else 0.0,
            "fpr": round(fpr, 4)
        }