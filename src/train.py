import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import re
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- NEU: SafeTensors für sicheres Speichern  ---
try:
    from safetensors.torch import save_model
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warnung: 'safetensors' nicht installiert. Speichern wird übersprungen.")

# Local Modules
from src.benchmark import PerformanceMonitor
from src.data_loader import load_and_standardize_data, load_url_data, encode_urls
from src.models import CNNModel, GPULogisticRegression, get_xgboost_model, get_svm_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIX: Globale Reproduzierbarkeit [cite: 200-204] ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- FIX: Worker Seeds für DataLoader [cite: 207] ---
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_metrics(y_true, y_pred_binary, y_scores):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        fpr = 0.0
    
    return {
        "accuracy": round(accuracy_score(y_true, y_pred_binary), 4),
        "precision": round(precision_score(y_true, y_pred_binary, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_binary, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred_binary, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_scores), 4) if len(np.unique(y_true)) > 1 else 0.0,
        "fpr": round(fpr, 4)
    }

def enforce_imbalance(X, y, target_ratio=1000):
    y = np.array(y)
    benign_indices = np.where(y == 0)[0]
    phishing_indices = np.where(y == 1)[0]
    
    n_benign = len(benign_indices)
    n_phishing_needed = int(n_benign / target_ratio)
    
    if n_phishing_needed < 1: n_phishing_needed = 1
        
    if len(phishing_indices) < n_phishing_needed:
        chosen_phishing = phishing_indices
    else:
        chosen_phishing = np.random.choice(phishing_indices, n_phishing_needed, replace=False)
        
    keep_indices = np.concatenate([benign_indices, chosen_phishing])
    np.random.shuffle(keep_indices)
    
    y_new = y[keep_indices]
    if isinstance(X, list):
        X_new = [X[i] for i in keep_indices]
    else:
        X_new = X[keep_indices]
        
    print(f"--- [Test Set] Imbalance Applied: {len(benign_indices)} Benign, {len(chosen_phishing)} Phishing ---")
    return X_new, y_new

def run_cnn(file_path, args):
    print("--- Mode: CNN (Raw URLs) ---")
    urls, labels = load_url_data(file_path)

    if len(urls) == 0: return

    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=args.seed)
    
    # 2. Imbalance on Test
    if args.imbalance_ratio > 0:
        print(f"Simulating real-world scenario ({args.imbalance_ratio}:1) on TEST DATA...")
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=args.imbalance_ratio)

    # Vectorization
    chars = sorted(list(set("".join(X_train))))
    char_to_int = {c: i+2 for i, c in enumerate(chars)}
    vocab_size = len(char_to_int) + 2
    
    print("Vectorizing URLs...")
    X_train_enc = torch.tensor(encode_urls(X_train, char_to_int), dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_enc = torch.tensor(encode_urls(X_test, char_to_int), dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # DataLoader Optimierung
    num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
    use_pin_memory = torch.cuda.is_available()
    
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        TensorDataset(X_train_enc, y_train_tensor), 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test_enc, y_test_tensor), 
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )

    model = CNNModel(vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss für Logits
    criterion = nn.BCEWithLogitsLoss() 
    
    # --- NEU: Mixed Precision Scaler [cite: 168-169] ---
    # Verhindert Gradient Underflow bei FP16 Training
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    
    monitor = PerformanceMonitor("CNN", dataset_name=os.path.basename(file_path))
    
    print("Starting Training (with AMP)...")
    monitor.start_measurement()
    
    for epoch in range(args.epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE, non_blocking=True), y_b.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # --- NEU: Autocast Context [cite: 166] ---
            # PyTorch wählt automatisch FP16 für Tensor Cores wo möglich
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
            
            # --- NEU: Scaled Backward Pass ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        print(f"Epoch {epoch+1} done.")
    
    monitor.end_measurement(task_name="Training")

    # --- NEU: Speichern mit SafeTensors  ---
    if SAFETENSORS_AVAILABLE:
        model_filename = f"cnn_model_seed{args.seed}.safetensors"
        save_path = os.path.join("results", model_filename) # Geht davon aus, dass 'results' existiert
        os.makedirs("results", exist_ok=True)
        save_model(model, save_path)
        print(f"Model saved securely to: {save_path}")

    # Inference
    print("Starting Inference...")
    monitor.start_measurement()
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(DEVICE, non_blocking=True)
            # Auch bei Inference kann AMP leicht helfen, meist aber weniger kritisch
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                outputs = model(X_b)
            
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.float().cpu().numpy())
            all_labels.extend(y_b.numpy())
    
    y_scores = np.array(all_preds)
    y_pred = (y_scores > 0.5).astype(int)
    metrics = calculate_metrics(np.array(all_labels), y_pred, y_scores)
    monitor.end_measurement(task_name="Inference", extra_metrics=metrics)

def run_svm_tfidf(file_path, args):
    print("--- Mode: SVM + TFIDF ---")
    urls, labels = load_url_data(file_path)
    text_data = [re.sub(r'\W+', ' ', str(u)) for u in urls]
    
    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=args.seed)
    
    if args.imbalance_ratio > 0:
        print(f"Simulating real-world scenario ({args.imbalance_ratio}:1) on TEST DATA...")
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=args.imbalance_ratio)
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    monitor = PerformanceMonitor("SVM_TFIDF", dataset_name=os.path.basename(file_path))
    model = get_svm_model()

    print("Starting Training...")
    monitor.start_measurement()
    model.fit(X_train_vec, y_train)
    monitor.end_measurement(task_name="Training")
    
    print("Starting Inference...")
    monitor.start_measurement()
    y_pred = model.predict(X_test_vec)
    y_scores = model.predict_proba(X_test_vec)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    monitor.end_measurement(task_name="Inference", extra_metrics=metrics)

def run_numeric_model(model_type, file_path, args):
    print(f"--- Mode: {model_type.upper()} ---")
    X, y = load_and_standardize_data(file_path, "label") 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    
    if hasattr(X_train, "values"): X_train = X_train.values
    if hasattr(y_train, "values"): y_train = y_train.values
    if hasattr(X_test, "values"): X_test = X_test.values
    if hasattr(y_test, "values"): y_test = y_test.values

    if args.imbalance_ratio > 0:
        print(f"Simulating real-world scenario ({args.imbalance_ratio}:1) on TEST DATA...")
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=args.imbalance_ratio)

    scaler_data = StandardScaler()
    if model_type == "lr":
        X_train = scaler_data.fit_transform(X_train)
        X_test = scaler_data.transform(X_test)
    
    monitor = PerformanceMonitor(model_type.upper(), dataset_name=os.path.basename(file_path))
    
    if model_type == "lr":
        model = GPULogisticRegression(X_train.shape[1], DEVICE)
        monitor.start_measurement()
        
        # Hinweis: Auch hier könnte man AMP einbauen, bringt bei kleinen LR-Modellen aber wenig Gewinn.
        model.fit(X_train, y_train, epochs=args.epochs, lr=args.lr)
        
        monitor.end_measurement(task_name="Training")
        
        monitor.start_measurement()
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = (y_scores > 0.5).astype(int)
        
    elif model_type == "xgb":
        model = get_xgboost_model(torch.cuda.is_available(), n_estimators=args.epochs, learning_rate=args.lr)
        monitor.start_measurement()
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=args.seed)
        
        model.fit(X_tr.astype(float), y_tr.astype(float),
                  eval_set=[(X_val.astype(float), y_val.astype(float))], verbose=False)
        monitor.end_measurement(task_name="Training")
        
        monitor.start_measurement()
        X_test_f = X_test.astype(float)
        y_scores = model.predict_proba(X_test_f)[:, 1]
        y_pred = model.predict(X_test_f)

    metrics = calculate_metrics(y_test, y_pred, y_scores)
    monitor.end_measurement(task_name="Inference", extra_metrics=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Benchmark")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "lr", "svm", "xgb"], help="Model selection")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs (NN) or Estimators (XGB)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--imbalance_ratio", type=int, default=1000, help="Test Set Ratio Benign:Phishing")
    
    args = parser.parse_args()
    seed_everything(args.seed)
    
    print(f"--- Config: {args.model.upper()} | Ratio: {args.imbalance_ratio}:1 | File: {args.file} ---")

    if args.model == "xgb":
        if args.epochs == 5: args.epochs = 100
        if args.lr == 0.001: args.lr = 0.1

    if args.model == "cnn":
        run_cnn(args.file, args) 
    elif args.model == "svm":
        run_svm_tfidf(args.file, args) 
    elif args.model in ["lr", "xgb"]:
        run_numeric_model(args.model, args.file, args)