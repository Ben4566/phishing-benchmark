import pandas as pd
import numpy as np
import os

def load_and_standardize_data(filepath, target_col_name="label"):
    print(f"--- [Loader] Starte Laden von: {os.path.basename(filepath)} ---")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except Exception as e:
        print(f"Warnung: Standard-Laden fehlgeschlagen ({e}), versuche Fallback...")
        df = pd.read_csv(filepath, error_bad_lines=False)

    # 1. Validate target variable existence
    # Since you standardized everything to "label", we strictly look for it.
    if target_col_name not in df.columns:
        raise ValueError(f"Error: The column '{target_col_name}' was not found in {filepath}.")

    real_target_col = target_col_name
    print(f"--- [Loader] Zielvariable identifiziert als: '{real_target_col}' ---")

    # 2. Extract y (target)
    y_raw = df[real_target_col]

    # 3. Prepare X: Drop target column
    X_raw = df.drop(columns=[real_target_col])

    # 4. Remove unnecessary metadata columns
    drop_candidates = ['FILENAME', 'filename', 'URL', 'url', 'id', 'ID', 'Index', 'index', 'uuid', 'Unnamed: 0']
    X_raw = X_raw.drop(columns=drop_candidates, errors='ignore')

    # 5. Force X to numeric
    # Coerce errors to NaN, then fill with 0
    X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 6. Process y
    if y_raw.dtype == 'object' or hasattr(y_raw, 'cat'):
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    print(f"--- [Loader] Fertig. Features: {X.shape[1]}, Samples: {X.shape[0]} ---")
    return X, y

# --- Function for Raw URLs (CNN, SVM) ---
def load_url_data(csv_file_path):
    print(f"Lade URL-Daten aus: {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
        
        # Determine columns (simplified)
        url_col = 'URL' if 'URL' in df.columns else 'url'
        label_col = 'label'  # Strictly use 'label' now
        
        # Check if required columns exist
        if url_col not in df.columns:
            print(f"Fehler: URL-Spalte '{url_col}' nicht gefunden.")
            return [], np.array([])
            
        if label_col not in df.columns:
            print(f"Fehler: Label-Spalte '{label_col}' nicht gefunden.")
            return [], np.array([])

        df = df.dropna(subset=[url_col, label_col])
        urls = df[url_col].astype(str).tolist()
        
        # Process labels
        if pd.api.types.is_numeric_dtype(df[label_col]):
            labels = df[label_col].values
        else:
            # Keep mapping just in case values are strings (e.g., 'phishing'/'benign')
            label_mapping = {'phishing': 1, 'bad': 1, 'malicious': 1, 
                             'benign': 0, 'good': 0, 'legitimate': 0}
            labels = df[label_col].map(label_mapping)
            
            # Filter out unmapped labels
            if labels.isna().any():
                mask = labels.notna()
                urls = [u for u, m in zip(urls, mask) if m]
                labels = labels.dropna()
            labels = labels.values.astype(int)
            
        print(f"Erfolgreich geladen: {len(urls)} URLs.")
        return urls, np.array(labels)

    except Exception as e:
        print(f"Fehler beim Laden der CSV: {e}")
        return [], np.array([])

def encode_urls(urls, char_to_int, max_len=200):
    encoded_batch = []
    for url in urls:
        vec = [char_to_int.get(c, 1) for c in url]
        if len(vec) < max_len:
            vec += [0] * (max_len - len(vec))
        else:
            vec = vec[:max_len]
        encoded_batch.append(vec)
    return np.array(encoded_batch)