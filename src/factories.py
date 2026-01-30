import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import DictConfig

# Importiere deine Modelle und Adapter
from src.models import CNNModel, GPULogisticRegression, get_xgboost_model, get_svm_model
from src.adapters import PyTorchAdapter, SklearnAdapter
from src.interfaces import IPredictor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model_adapter(cfg: DictConfig, input_dim: int = 0, vocab_size: int = 0) -> IPredictor:
    """
    Factory-Methode: Erstellt basierend auf der Config das richtige Modell
    und verpackt es in den passenden Adapter.
    """
    model_name = cfg.model.name.lower()

    if model_name == "cnn":
        # PyTorch Setup
        net = CNNModel(vocab_size=vocab_size, embed_dim=32).to(DEVICE)
        optimizer = optim.Adam(net.parameters(), lr=cfg.model.params.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        return PyTorchAdapter(
            model=net,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            batch_size=cfg.model.params.batch_size,
            epochs=cfg.model.params.epochs,
            seed=cfg.seed
        )

    elif model_name == "xgb":
        # XGBoost Setup
        xgb_model = get_xgboost_model(
            use_cuda=torch.cuda.is_available(),
            n_estimators=cfg.model.params.epochs, # Wir nutzen epochs als n_estimators
            learning_rate=cfg.model.params.lr
        )
        # Injection: Wir sagen dem Adapter, er soll Eval-Sets nutzen
        return SklearnAdapter(xgb_model, use_eval_set=True)

    elif model_name == "svm":
        # NEU: Wir bauen eine Pipeline aus Vektorisierer + SVM.
        # Das Modell akzeptiert dann direkt Listen von Strings (URLs).
        base_svm = get_svm_model() # Liefert CalibratedClassifierCV(LinearSVC)
        
        # Pipeline erstellen: Erst TF-IDF, dann SVM
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)), # Optional: Features begrenzen
            ('clf', base_svm)
        ])
        
        return SklearnAdapter(model_pipeline)

# Ausschnitt aus src/factories.py
    elif model_name == "lr":
        lr_model = GPULogisticRegression(
            input_dim, 
            DEVICE,
            epochs=cfg.model.params.epochs,  # <--- Jetzt hier übergeben!
            lr=cfg.model.params.lr,
            batch_size=cfg.model.params.batch_size
        )
        return SklearnAdapter(lr_model)

    else:
        raise ValueError(f"Unknown model: {model_name}")