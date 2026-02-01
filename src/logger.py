import logging
import os
import sys
from datetime import datetime

# Configuration
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Environment Variable Key for IPC (Inter-Process Communication)
ENV_LOG_FILE_KEY = "BENCHMARK_LOG_FILE"

def setup_logger(name=__name__, log_file=None, level=logging.INFO):
    """
    Initializes a thread-safe / process-aware logger configuration.
    
    Mechanism:
    Uses `os.environ` to broadcast the active log filename to child processes 
    (e.g., PyTorch Dataloaders or Hydra workers), ensuring all logs from a single 
    execution run are aggregated into one file.
    
    Args:
        name (str): Logger namespace.
        log_file (str, optional): Explicit filename override.
        level (int): Logging threshold.
    """
    
    # 1. Resolve Log Filename
    if log_file is None:
        # Check if a parent process has already established a session
        if ENV_LOG_FILE_KEY in os.environ:
            target_log_file = os.environ[ENV_LOG_FILE_KEY]
        else:
            # Initialize new session: Generate timestamped file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_log_file = os.path.join(LOG_DIR, f"benchmark_run_{timestamp}.log")
            os.environ[ENV_LOG_FILE_KEY] = target_log_file
    else:
        # Explicit override
        target_log_file = os.path.join(LOG_DIR, log_file)

    # 2. Define Output Format
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 3. Handler Configuration (Idempotency Check)
    if not logger.hasHandlers():
        # Handler A: File Output
        try:
            file_handler = logging.FileHandler(target_log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as e:
            # Graceful degradation: If file locking fails (common in Windows multiprocessing),
            # we rely on stdout.
            print(f"WARNING: Failed to bind log file (possible multiprocessing race condition): {e}")

        # Handler B: Console Output (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # Prevent log duplication in root logger
    logger.propagate = False

    return logger