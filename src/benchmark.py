import time
import torch
import psutil
import os
import json
import gc
import threading
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Versuche pynvml zu importieren (für hardware-nahe GPU-Messung)
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warnung: 'pynvml' nicht installiert. Hardware-VRAM (für XGBoost nötig) kann nicht gemessen werden.")

class PerformanceMonitor:
    def __init__(self, model_name, save_path="benchmark_results.json"):
        self.model_name = model_name
        self.save_path = save_path
        self.process = psutil.Process(os.getpid())
        self.gpu_load_history = []
        self.max_vram_system = 0  # Maximaler System-VRAM (für XGBoost & Torch Cache)
        self.monitoring_active = False

    def _monitor_gpu_usage(self):
        """Hintergrund-Thread: Überwacht die echte Hardware-Last (für alle Frameworks)"""
        while self.monitoring_active:
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # 1. Auslastung (%)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_load_history.append(util.gpu)
                    
                    # 2. Echter VRAM Verbrauch (inkl. Cache, XGBoost, etc.)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_vram_mb = mem_info.used / (1024 ** 2)
                    
                    if current_vram_mb > self.max_vram_system:
                        self.max_vram_system = current_vram_mb
                        
                except Exception:
                    self.gpu_load_history.append(0)
            else:
                self.gpu_load_history.append(0)
            
            time.sleep(0.1) 

    def start_measurement(self):
        # 1. Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # 2. Reset Stats
        self.gpu_load_history = []
        self.max_vram_system = 0 
        
        # Falls pynvml nicht geht, Initialwert setzen damit nicht 0 bleibt
        if PYNVML_AVAILABLE:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.max_vram_system = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**2)
            except: pass

        self.process.cpu_percent(interval=None)
        self.monitoring_active = True
        
        # 3. Start Monitor Thread
        self.monitor_thread = threading.Thread(target=self._monitor_gpu_usage)
        self.monitor_thread.start()
        
        # 4. Zeit Start
        self.start_time = time.time()

    def end_measurement(self, task_name="inference", extra_metrics=None):
        # 1. Stop Monitor
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.monitoring_active = False
        self.monitor_thread.join()
        
        end_time = time.time()
        
        # 2. Basis-Daten berechnen
        duration = end_time - self.start_time
        ram_usage = self.process.memory_info().rss / (1024 ** 2) # System RAM
        cpu_usage = self.process.cpu_percent(interval=None)
        
        # 3. VRAM Auswertung
        # A) System VRAM (Das "Echte" Limit, relevant für XGBoost & OOM)
        vram_peak_system = self.max_vram_system
        
        # B) Torch VRAM (Spezifisch für PyTorch Tensoren)
        vram_peak_torch = 0
        if torch.cuda.is_available():
            vram_peak_torch = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Fallback: Wenn pynvml fehlte, nutze Torch-Wert als System-Wert
        if vram_peak_system == 0 and vram_peak_torch > 0:
            vram_peak_system = vram_peak_torch

        avg_gpu_util = 0
        if self.gpu_load_history:
            avg_gpu_util = sum(self.gpu_load_history) / len(self.gpu_load_history)

        # 4. Speichern
        data = {
            "model": self.model_name,
            "task": task_name,
            "time_sec": round(duration, 4),
            "ram_mb": round(ram_usage, 2),
            "vram_mb": round(vram_peak_system, 2),       # Hardware-Verbrauch (Alles)
            "torch_vram_mb": round(vram_peak_torch, 2),  # Nur PyTorch-Tensoren
            "cpu_percent": round(cpu_usage, 1),
            "gpu_util_percent": round(avg_gpu_util, 1)
        }
        
        if extra_metrics:
            data.update(extra_metrics)
        
        print(f"--- Ergebnisse {self.model_name} ({task_name}) ---")
        print(f"Zeit: {data['time_sec']}s | GPU-Last: {data['gpu_util_percent']}%")
        print(f"VRAM (System): {data['vram_mb']} MB | VRAM (Torch): {data['torch_vram_mb']} MB")
        
        self._save_to_file(data)
        return data

def calculate_metrics(y_true, y_pred, y_scores=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_scores), 4) if y_scores is not None else 0,
        "fpr": round(fpr, 4)
    }


    def _save_to_file(self, data):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                try:
                    history = json.load(f)
                except:
                    history = []
        else:
            history = []
        history.append(data)
        with open(self.save_path, 'w') as f:
            json.dump(history, f, indent=4)