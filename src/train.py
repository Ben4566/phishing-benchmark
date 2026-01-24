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

# Lokale Module
from benchmark import PerformanceMonitor
from data_loader import load_and_standardize_data, load_url_data, encode_urls
from models import CNNModel, GPULogisticRegression, get_xgboost_model, get_svm_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred_binary, y_scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "accuracy": round(accuracy_score(y_true, y_pred_binary), 4),
        "precision": round(precision_score(y_true, y_pred_binary, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_binary, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred_binary, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_scores), 4),
        "fpr": round(fpr, 4)
    }

def run_cnn(file_path, args):
    print("--- Modus: CNN (Raw URLs) ---")
    urls, labels = load_url_data(file_path)

    if len(urls) == 0: return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=42)
    
    # Vectorization Logic
    chars = sorted(list(set("".join(X_train))))
    char_to_int = {c: i+2 for i, c in enumerate(chars)}
    vocab_size = len(char_to_int) + 2
    
    print("Vektorisiere URLs...")
    X_train_enc = torch.tensor(encode_urls(X_train, char_to_int), dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
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
    print("Starte Training...")
    monitor.start_measurement()
    model.train()
    for epoch in range(args.epochs): # Epochs fest codiert oder via args übergeben
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done.")
    monitor.end_measurement(task_name="Training")

    # Inference
    print("Starte Inferenz...")
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
    monitor.end_measurement(task_name="Inferenz", extra_metrics=metrics)

def run_svm_tfidf(file_path, args):
    print("--- Modus: SVM + TFIDF (Raw URLs) ---")
    urls, labels = load_url_data(file_path)
    
    # Text Preprocessing (aus SVM Notebook)
    text_data = [re.sub(r'\W+', ' ', str(u)) for u in urls]
    
    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.1, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    monitor = PerformanceMonitor("SVM_TFIDF", dataset_name=os.path.basename(file_path))
    model = get_svm_model()

    print("Starte Training...")
    monitor.start_measurement()
    model.fit(X_train_vec, y_train)
    monitor.end_measurement(task_name="Training")
    
    print("Starte Inferenz...")
    monitor.start_measurement()
    y_pred = model.predict(X_test_vec)
    y_scores = model.predict_proba(X_test_vec)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    monitor.end_measurement(task_name="Inferenz", extra_metrics=metrics)

def run_numeric_model(model_type, file_path, args):
    print(f"--- Modus: {model_type.upper()} (Numeric Features) ---")
    # Features laden (standardisiert)
    X, y = load_and_standardize_data(file_path, "label") 
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Skalierung (Wichtig für LogReg)
    if model_type == "lr":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
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
        model.fit(X_train.astype(float), y_train.astype(float),
                  eval_set=[(X_val.astype(float), y_val.astype(float))], verbose=False)
        monitor.end_measurement(task_name="Training")
        
        monitor.start_measurement()
        X_test_f = X_test.astype(float)
        y_scores = model.predict_proba(X_test_f)[:, 1]
        y_pred = model.predict(X_test_f)

    # Metrics
    if hasattr(y_test, "values"): y_test = y_test.values
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    monitor.end_measurement(task_name="Inferenz", extra_metrics=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Benchmark")
    
    # Pflichtfelder
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "lr", "svm", "xgb"], help="Wahl des Modells")
    parser.add_argument("--file", type=str, required=True, help="Pfad zur CSV-Datei")
    
    # Optionale Hyperparameter (mit Standardwerten für Neuronale Netze)
    parser.add_argument("--epochs", type=int, default=5, help="Anzahl der Epochen (NN) oder Bäume (XGB)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    
    args = parser.parse_args()
    
    print(f"--- Initiale Config: {args.model.upper()} | Epochs: {args.epochs} | LR: {args.lr} ---")

    # --- SPEZIAL-LOGIK FÜR XGBOOST ---
    # Falls XGBoost gewählt wurde und noch die "schlechten" Defaults für CNN drinstehen (5 Epochen, 0.001 LR),
    # passen wir sie automatisch auf sinnvolle XGBoost-Werte an.
    if args.model == "xgb":
        # Wenn User 'epochs' nicht geändert hat (steht auf 5), setzen wir es auf 100 Bäume
        if args.epochs == 5:
            print("-> Auto-Fix: Setze 'epochs' (n_estimators) für XGBoost von 5 auf 100 hoch.")
            args.epochs = 100
        
        # Wenn User 'lr' nicht geändert hat (steht auf 0.001), setzen wir es auf 0.1
        if args.lr == 0.001:
            print("-> Auto-Fix: Setze 'lr' (learning_rate) für XGBoost von 0.001 auf 0.1 hoch.")
            args.lr = 0.1

    # --- RUN LOGIC ---
    if args.model == "cnn":
        # Korrektur: Übergibt args, damit batch_size/lr genutzt werden können
        run_cnn(args.file, args) 
        
    elif args.model == "svm":
        # Korrektur: Übergibt args (vorher fehlte das und führte zum Absturz)
        run_svm_tfidf(args.file, args) 
        
    elif args.model in ["lr", "xgb"]:
        # Übergibt args für dynamische Epochen/LR
        run_numeric_model(args.model, args.file, args)