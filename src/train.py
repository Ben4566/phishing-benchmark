import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Local Modules
from benchmark import PerformanceMonitor
from data_loader import load_and_standardize_data, load_url_data, encode_urls
from models import CNNModel, GPULogisticRegression, get_xgboost_model, get_svm_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred_binary, y_scores):
    """
    Calculates standard metrics for evaluation.
    """
    # Handle edge case where confusion matrix might be smaller if one class is missing
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        fpr = 0.0
    
    # Precision/Recall handle zero_division automatically
    return {
        "accuracy": round(accuracy_score(y_true, y_pred_binary), 4),
        "precision": round(precision_score(y_true, y_pred_binary, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_binary, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred_binary, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_scores), 4) if len(np.unique(y_true)) > 1 else 0.0,
        "fpr": round(fpr, 4)
    }

def enforce_imbalance(X, y, target_ratio=1000):
    """
    Applied to TEST SET only.
    Adjusts the data to have a specific ratio of Benign(0):Phishing(1).
    Example: target_ratio=1000 means 1000 benign samples for 1 phishing sample.
    """
    y = np.array(y)
    
    # Identify indices
    benign_indices = np.where(y == 0)[0]
    phishing_indices = np.where(y == 1)[0]
    
    n_benign = len(benign_indices)
    
    # Calculate required number of phishing samples: n_benign / target_ratio
    # e.g. 50,000 benign / 1000 = 50 phishing samples
    n_phishing_needed = int(n_benign / target_ratio)
    
    # Safety check: We need at least 1 positive sample to calculate recall
    if n_phishing_needed < 1:
        print(f"Warning: Test set too small for strictly {target_ratio}:1. Keeping 1 phishing sample.")
        n_phishing_needed = 1
        
    if len(phishing_indices) < n_phishing_needed:
        print(f"Warning: Not enough phishing samples in test set for ratio {target_ratio}:1. Using all available.")
        chosen_phishing = phishing_indices
    else:
        # Randomly select the reduced number of phishing samples
        chosen_phishing = np.random.choice(phishing_indices, n_phishing_needed, replace=False)
        
    # Combine and Shuffle
    keep_indices = np.concatenate([benign_indices, chosen_phishing])
    np.random.shuffle(keep_indices)
    
    # Filter y
    y_new = y[keep_indices]
    
    # Filter X (Handle both List (URLs) and Numpy Array (Features))
    if isinstance(X, list):
        X_new = [X[i] for i in keep_indices]
    else:
        # Assuming numpy array or similar sliceable object
        X_new = X[keep_indices]
        
    print(f"--- [Test Set] Imbalance Applied: {len(benign_indices)} Benign, {len(chosen_phishing)} Phishing ---")
    print(f"--- [Test Set] Real-World Ratio: {len(benign_indices)/len(chosen_phishing):.1f}:1 ---")
    
    return X_new, y_new

def run_cnn(file_path, args):
    print("--- Mode: CNN (Raw URLs) ---")
    urls, labels = load_url_data(file_path)

    if len(urls) == 0: return

    # 1. Split Data (80% Train, 20% Test)
    # Train set remains balanced (or as is) so the model learns well.
    X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=args.seed)
    
    # 2. Apply Imbalance ONLY to Test Data (Real World Simulation)
    if args.imbalance_ratio > 0:
        print(f"Simulating real-world scenario ({args.imbalance_ratio}:1) on TEST DATA...")
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=args.imbalance_ratio)

    # Vectorization Logic
    # Build vocab based on Training data
    chars = sorted(list(set("".join(X_train))))
    char_to_int = {c: i+2 for i, c in enumerate(chars)}
    vocab_size = len(char_to_int) + 2
    
    print("Vectorizing URLs...")
    X_train_enc = torch.tensor(encode_urls(X_train, char_to_int), dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # Encode Test data (using training vocab)
    X_test_enc = torch.tensor(encode_urls(X_test, char_to_int), dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_enc, y_train_tensor), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_enc, y_test_tensor), batch_size=args.batch_size)

    # Model Setup
    model = CNNModel(vocab_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss() 
    monitor = PerformanceMonitor("CNN", dataset_name=os.path.basename(file_path))
    
    # Training
    print("Starting Training...")
    monitor.start_measurement()
    model.train()
    for epoch in range(args.epochs):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done.")
    monitor.end_measurement(task_name="Training")

    # Inference (on the imbalanced 1000:1 test set)
    print("Starting Inference (Real World Simulation)...")
    monitor.start_measurement()
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            outputs = model(X_b.to(DEVICE))
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_b.numpy())
    
    y_scores = np.array(all_preds)
    y_pred = (y_scores > 0.5).astype(int)
    metrics = calculate_metrics(np.array(all_labels), y_pred, y_scores)
    monitor.end_measurement(task_name="Inference", extra_metrics=metrics)

def run_svm_tfidf(file_path, args):
    print("--- Mode: SVM + TFIDF (Raw URLs) ---")
    urls, labels = load_url_data(file_path)
    
    # Text Preprocessing
    text_data = [re.sub(r'\W+', ' ', str(u)) for u in urls]
    
    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=args.seed)
    
    # 2. Apply Imbalance ONLY to Test Data
    if args.imbalance_ratio > 0:
        print(f"Simulating real-world scenario ({args.imbalance_ratio}:1) on TEST DATA...")
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=args.imbalance_ratio)
    
    vectorizer = TfidfVectorizer()
    # Fit only on training data to avoid data leakage
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    monitor = PerformanceMonitor("SVM_TFIDF", dataset_name=os.path.basename(file_path))
    model = get_svm_model()

    print("Starting Training...")
    monitor.start_measurement()
    model.fit(X_train_vec, y_train)
    monitor.end_measurement(task_name="Training")
    
    print("Starting Inference (Real World Simulation)...")
    monitor.start_measurement()
    y_pred = model.predict(X_test_vec)
    y_scores = model.predict_proba(X_test_vec)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    monitor.end_measurement(task_name="Inference", extra_metrics=metrics)

def run_numeric_model(model_type, file_path, args):
    print(f"--- Mode: {model_type.upper()} (Numeric Features) ---")
    # Features laden (standardisiert)
    X, y = load_and_standardize_data(file_path, "label") 
    
    # 1. Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    # Further split temp into Val/Test if needed, or just treat temp as test.
    # Let's keep it consistent: Train on 70%, Test on 30% (simplified from previous logic)
    # Or strict 80/20 as requested:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    
    # Ensure numpy format
    if hasattr(X_train, "values"): X_train = X_train.values
    if hasattr(y_train, "values"): y_train = y_train.values
    if hasattr(X_test, "values"): X_test = X_test.values
    if hasattr(y_test, "values"): y_test = y_test.values

    # 2. Apply Imbalance ONLY to Test Data
    if args.imbalance_ratio > 0:
        print(f"Simulating real-world scenario ({args.imbalance_ratio}:1) on TEST DATA...")
        X_test, y_test = enforce_imbalance(X_test, y_test, target_ratio=args.imbalance_ratio)

    # Scaling (Important for LogReg)
    scaler = StandardScaler()
    if model_type == "lr":
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    monitor = PerformanceMonitor(model_type.upper(), dataset_name=os.path.basename(file_path))
    
    if model_type == "lr":
        model = GPULogisticRegression(X_train.shape[1], DEVICE)
        monitor.start_measurement()
        model.fit(X_train, y_train, epochs=args.epochs, lr=args.lr)
        monitor.end_measurement(task_name="Training")
        
        monitor.start_measurement()
        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred = (y_scores > 0.5).astype(int)
        
    elif model_type == "xgb":
        model = get_xgboost_model(torch.cuda.is_available(), n_estimators=args.epochs, learning_rate=args.lr)
        monitor.start_measurement()
        # Create a small validation set from training data strictly for XGBoost early stopping (optional)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=args.seed)
        
        model.fit(X_tr.astype(float), y_tr.astype(float),
                  eval_set=[(X_val.astype(float), y_val.astype(float))], verbose=False)
        monitor.end_measurement(task_name="Training")
        
        monitor.start_measurement()
        X_test_f = X_test.astype(float)
        y_scores = model.predict_proba(X_test_f)[:, 1]
        y_pred = model.predict(X_test_f)

    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    monitor.end_measurement(task_name="Inference", extra_metrics=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Benchmark")
    
    # Required Arguments
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "lr", "svm", "xgb"], help="Model selection")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    
    # Optional Hyperparameters
    parser.add_argument("--epochs", type=int, default=5, help="Epochs (NN) or Estimators (XGB)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed für Reproduzierbarkeit")
    # Imbalance Ratio for TEST set
    parser.add_argument("--imbalance_ratio", type=int, default=1000, help="Test Set Ratio Benign:Phishing (e.g. 1000). 0 to disable.")
    
    args = parser.parse_args()
    
    print(f"--- Config: {args.model.upper()} | Test-Set Ratio: {args.imbalance_ratio}:1 | File: {args.file} ---")

    # --- AUTO-CONFIG FOR XGBOOST ---
    if args.model == "xgb":
        if args.epochs == 5:
            print("-> Auto-Fix: Increasing 'epochs' (n_estimators) for XGBoost from 5 to 100.")
            args.epochs = 100
        if args.lr == 0.001:
            print("-> Auto-Fix: Increasing 'lr' (learning_rate) for XGBoost from 0.001 to 0.1.")
            args.lr = 0.1

    # --- RUN LOGIC ---
    if args.model == "cnn":
        run_cnn(args.file, args) 
        
    elif args.model == "svm":
        run_svm_tfidf(args.file, args) 
        
    elif args.model in ["lr", "xgb"]:
        run_numeric_model(args.model, args.file, args)