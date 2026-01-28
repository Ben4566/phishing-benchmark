import time
import torch
import psutil
import os
import json
import gc
import threading
import numpy as np
from typing import Optional, Dict, Any

# Hydra importieren um das Original-Verzeichnis zu finden
import hydra 

# --- LOGGING INTEGRATION ---
from src.logger import setup_logger
logger = setup_logger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("'pynvml' bindings not found. High-precision VRAM monitoring disabled.")

class PerformanceMonitor:
    def __init__(self, model_name: str, dataset_name: str = "unknown", save_path: str = "benchmark_results.json", sampling_rate: float = 0.01):
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # --- FIX FÜR HYDRA ---
        # Hydra ändert das Arbeitsverzeichnis. Wir wollen aber im Hauptprojekt speichern.
        try:
            # Holt den Pfad, wo du den Befehl ausgeführt hast
            orig_cwd = hydra.utils.get_original_cwd()
            self.save_path = os.path.join(orig_cwd, save_path)
        except Exception:
            # Fallback, falls Hydra nicht läuft
            self.save_path = save_path
            
        self.process = psutil.Process(os.getpid())
        self.sampling_rate = sampling_rate
        
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        self.monitoring_active = False

    def _monitor_gpu_usage(self):
        while self.monitoring_active:
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    self.gpu_load_history.append(util)
                    if util > self.peak_gpu_util:
                        self.peak_gpu_util = util
                    
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_vram_mb = mem_info.used / (1024 ** 2)
                    if current_vram_mb > self.max_vram_system:
                        self.max_vram_system = current_vram_mb
                except Exception:
                    pass
            time.sleep(self.sampling_rate) 

    def start_measurement(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        
        if PYNVML_AVAILABLE:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.max_vram_system = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**2)
            except: pass

        self.start_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        self.start_time = time.perf_counter()
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu_usage)
        self.monitor_thread.start()

    def end_measurement(self, task_name: str = "inference", extra_metrics: Optional[Dict[str, Any]] = None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.monitoring_active = False
        self.monitor_thread.join()
        
        end_time = time.perf_counter()
        end_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        
        duration = end_time - self.start_time
        cpu_usage_precise = ((end_cpu_time - self.start_cpu_time) / duration) * 100 if duration > 0 else 0
        ram_usage = self.process.memory_info().rss / (1024 ** 2)
        
        vram_peak_torch = 0
        if torch.cuda.is_available():
            vram_peak_torch = torch.cuda.max_memory_allocated() / (1024 ** 2)

        avg_gpu_util = np.mean(self.gpu_load_history) if self.gpu_load_history else 0

        data = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "task": task_name,
            "time_sec": round(duration, 4),
            "ram_mb": round(ram_usage, 2),
            "vram_system_peak_mb": round(self.max_vram_system, 2),
            "vram_torch_peak_mb": round(vram_peak_torch, 2),
            "cpu_percent_avg": round(cpu_usage_precise, 1),
            "gpu_util_percent_avg": round(avg_gpu_util, 1),
            "gpu_util_percent_peak": round(self.peak_gpu_util, 1)
        }
        
        if extra_metrics:
            data.update(extra_metrics)
        
        logger.info(f"--- Results {self.model_name} | Data: {self.dataset_name} ({task_name}) ---")
        # Speichert jetzt in die Datei im Hauptverzeichnis (Thread-safe genug für Sequential runs)
        self._save_to_file(data)
        return data

    def _save_to_file(self, data):
        # File Locking wäre hier ideal, aber für sequentielle Hydra Runs reicht append meistens.
        # Bei Multiprocessing (-m mit launcher=joblib) bräuchte man FileLock.
        history = []
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    history = json.load(f)
            except: pass
        
        history.append(data)
        
        with open(self.save_path, 'w') as f:
            json.dump(history, f, indent=4)