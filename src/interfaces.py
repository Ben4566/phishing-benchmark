from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, Tuple

class IPredictor(ABC):
    """
    Abstract Base Class (Interface) representing the Strategy Pattern for ML models.
    
    Architectural Role:
    Enforces the Dependency Inversion Principle (DIP). High-level benchmarking modules
    depend on this abstraction rather than concrete implementations (e.g., PyTorch vs Sklearn),
    allowing seamless interchangeability of algorithms.
    """

    @abstractmethod
    def train(self, 
              X_train: Union[np.ndarray, object], 
              y_train: Union[np.ndarray, object], 
              X_val: Optional[Union[np.ndarray, object]] = None, 
              y_val: Optional[Union[np.ndarray, object]] = None) -> None:
        """
        Executes the training routine.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val (optional): Validation features for early stopping / hyperparameter tuning.
            y_val (optional): Validation labels.
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, object]) -> np.ndarray:
        """
        Generates hard class labels (0 or 1).
        
        Returns:
            np.ndarray: Binary predictions with shape (N,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, object]) -> np.ndarray:
        """
        Estimates the probability of the positive class (Phishing).
        
        Returns:
            np.ndarray: Flattened array of floats [0.0, 1.0] with shape (N,).
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persists the model state to the filesystem.
        """
        pass