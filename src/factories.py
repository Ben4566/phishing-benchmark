import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import DictConfig

# Domain-specific imports
from src.models import CNNModel, GPULogisticRegression, get_xgboost_model, get_svm_model
from src.adapters import PyTorchAdapter, SklearnAdapter
from src.interfaces import IPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model_adapter(cfg: DictConfig, input_dim: int = 0, vocab_size: int = 0) -> IPredictor:
    """
    Factory Method: Instantiates and configures the appropriate model adapter.
    
    Architectural Role:
    Decouples the configuration logic (Hydra/YAML) from the object instantiation.
    Ensures that the rest of the application interacts only with the 'IPredictor' interface,
    unaware of the underlying framework (PyTorch vs Scikit-Learn vs XGBoost).
    
    Args:
        cfg: The configuration node containing model hyperparameters.
        input_dim: Feature dimensionality (required for Logistic Regression).
        vocab_size: Vocabulary size (required for CNN embeddings).
    
    Returns:
        IPredictor: A unified interface wrapper around the specific model.
    """
    model_name = cfg.model.name.lower()

    if model_name == "cnn":
        # Deep Learning Path (PyTorch)
        net = CNNModel(vocab_size=vocab_size, embed_dim=32).to(DEVICE)
        
        # Optimizer Configuration
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
        # Gradient Boosting Path (XGBoost)
        xgb_model = get_xgboost_model(
            use_cuda=torch.cuda.is_available(),
            # Mapping generic 'epochs' config to tree-specific 'n_estimators'
            n_estimators=cfg.model.params.epochs, 
            learning_rate=cfg.model.params.lr
        )
        # Enable 'use_eval_set' to leverage validation data for early stopping inside the adapter
        return SklearnAdapter(xgb_model, use_eval_set=True)

    elif model_name == "svm":
        # Support Vector Machine Path (Scikit-Learn)
        # Architectural Decision: 
        # The TfidfVectorizer is deliberately excluded here. Vectorization is moved 
        # to the data loading phase ('run_benchmark.py') to ensure the benchmark 
        # measures purely the SVM training time, not text preprocessing.
        base_svm = get_svm_model() 
        return SklearnAdapter(base_svm)

    elif model_name == "lr":
        # Custom GPU-Accelerated Logistic Regression
        lr_model = GPULogisticRegression(
            input_dim, 
            DEVICE,
            epochs=cfg.model.params.epochs,
            lr=cfg.model.params.lr,
            batch_size=cfg.model.params.batch_size
        )
        return SklearnAdapter(lr_model)

    else:
        raise ValueError(f"Configuration Error: Unknown model identifier '{model_name}'")