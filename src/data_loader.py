# Datei: data_loader.py
import pandas as pd
import numpy as np
import os

def load_and_standardize_data(filepath, target_col_name, delimiter=","):
    """
    Lädt einen Datensatz, entfernt Metadaten/IDs und gibt X (Features) und y (Target) zurück.
    Garantiert identisches Preprocessing für alle Notebooks.
    """
    print(f"--- [Loader] Starte Laden von: {os.path.basename(filepath)} ---")
    
    # 1. Laden (Robust gegen 'bad lines')
    try:
        df = pd.read_csv(filepath, delimiter=delimiter, on_bad_lines='skip')
    except TypeError:
        # Fallback für ältere Pandas Versionen
        df = pd.read_csv(filepath, delimiter=delimiter, error_bad_lines=False)

    # 2. Sicherheits-Drop: IDs und Metadaten entfernen
    # Diese Liste enthält alles, was kein echtes Feature ist
    drop_candidates = [
        'FILENAME', 'filename', 
        'URL', 'url', 
        'id', 'ID', 'Index', 'index', 'uuid', 
        'Unnamed: 0'
    ]
    # 'errors=ignore' sorgt dafür, dass es nicht abstürzt, wenn eine Spalte fehlt
    df = df.drop(columns=drop_candidates, errors='ignore')
    
    # 3. Target-Spalte finden und retten
    # Falls der übergebene Name nicht stimmt, suchen wir ihn oder nehmen die letzte Spalte
    real_target_col = None
    
    if target_col_name in df.columns:
        real_target_col = target_col_name
    else:
        print(f"Warnung: Target '{target_col_name}' nicht gefunden. Versuche Alternativen...")
        # Liste typischer Target-Namen
        alternatives = ['label', 'class', 'Phishing?', 'phishing', 'target']
        for alt in alternatives:
            if alt in df.columns:
                real_target_col = alt
                print(f"-> Habe '{alt}' als Target identifiziert.")
                break
        
        # Wenn immer noch nichts gefunden, nimm die letzte Spalte
        if real_target_col is None:
            real_target_col = df.columns[-1]
            print(f"-> Fallback: Nutze letzte Spalte '{real_target_col}' als Target.")

    # Target temporär sichern
    y_raw = df[real_target_col]
    
    # 4. Nur Zahlen behalten (Der "Fairness-Filter")
    # Wirft alles raus, was Text ist (Domain-Namen, Pfade etc.)
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 5. X und y finalisieren
    # Target aus X entfernen (falls es noch drin ist)
    if real_target_col in df_numeric.columns:
        X = df_numeric.drop(columns=[real_target_col])
    else:
        # Target war eh Text und wurde schon durch select_dtypes entfernt
        X = df_numeric 
        
    # Target numerisch machen (für Modelle)
    if y_raw.dtype == 'object' or hasattr(y_raw, 'cat'):
        print("-> Wandle Text-Labels in Zahlen um...")
        y = y_raw.astype('category').cat.codes
    else:
        y = y_raw

    print(f"--- [Loader] Fertig. Features: {X.shape[1]}, Samples: {X.shape[0]} ---")
    return X, y