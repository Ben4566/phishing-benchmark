import logging
import os
import sys
from datetime import datetime

# Konfiguration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Konstante für die Umgebungsvariable
ENV_LOG_FILE_KEY = "BENCHMARK_LOG_FILE"

def setup_logger(name=__name__, log_file=None, level=logging.INFO):
    """
    Erstellt einen konfigurierten Logger.
    Verwendet os.environ, um sicherzustellen, dass Multiprocessing-Worker
    in dieselbe Datei schreiben.
    """
    
    # 1. Dateinamen bestimmen
    if log_file is None:
        # Prüfen, ob eine übergeordnete Instanz (Hauptprozess) schon einen Namen gesetzt hat
        if ENV_LOG_FILE_KEY in os.environ:
            target_log_file = os.environ[ENV_LOG_FILE_KEY]
        else:
            # Neuer Lauf: Namen generieren und in Environment setzen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_log_file = os.path.join(LOG_DIR, f"benchmark_run_{timestamp}.log")
            os.environ[ENV_LOG_FILE_KEY] = target_log_file
    else:
        # Expliziter Name (überschreibt Automatik)
        target_log_file = os.path.join(LOG_DIR, log_file)

    # Formatierung definieren
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # WICHTIG: Verhindern, dass Handler mehrfach hinzugefügt werden
    if not logger.hasHandlers():
        # Handler 1: Datei
        # Wir fangen Fehler ab, falls mehrere Prozesse gleichzeitig die Datei öffnen (Windows-Locking)
        try:
            file_handler = logging.FileHandler(target_log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as e:
            # Im Worker-Prozess ist es oft nicht kritisch, wenn das File-Logging fehlschlägt,
            # solange der Hauptprozess schreibt.
            print(f"WARNUNG: Konnte Log-Datei nicht binden (evtl. Multiprocessing Konflikt): {e}")

        # Handler 2: Konsole (immer sicher)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # Verhindert, dass Logs doppelt an den Root-Logger geschickt werden
    logger.propagate = False

    return logger