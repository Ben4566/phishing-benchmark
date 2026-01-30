from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, Tuple

class IPredictor(ABC):
    """
    Abstraktes Interface (Contract) für alle ML-Modelle.
    Dient der Einhaltung des Dependency Inversion Principle (DIP).
    """

    @abstractmethod
    def train(self, 
              X_train: Union[np.ndarray, object], 
              y_train: Union[np.ndarray, object], 
              X_val: Optional[Union[np.ndarray, object]] = None, 
              y_val: Optional[Union[np.ndarray, object]] = None) -> None:
        """
        Trainiert das Modell.
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, object]) -> np.ndarray:
        """
        Gibt binäre Vorhersagen (0 oder 1) zurück.
        Shape: (N,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, object]) -> np.ndarray:
        """
        Gibt Wahrscheinlichkeiten für die positive Klasse (1) zurück.
        Shape: (N,) - Achtung: Flattened array von Floats zwischen 0.0 und 1.0.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Speichert das Modell persistent.
        """
        pass