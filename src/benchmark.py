import time
import torch
import psutil
import os
import json
import gc
import threading
import numpy as np
from typing import Optional, Dict, Any
import hydra 

# --- LOGGING INTEGRATION ---
from src.logger import setup_logger
logger = setup_logger(__name__)

# --- OPTIONAL DEPENDENCY: GPU HARDWARE MONITORING ---
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("Optional dependency 'pynvml' not found. High-precision hardware VRAM monitoring is disabled.")
except Exception as e:
    PYNVML_AVAILABLE = False
    logger.warning(f"Failed to initialize NVML: {e}")

class PerformanceMonitor:
    """
    Resource Profiling Engine.
    
    Responsibilities:
    1. Tracks execution time (Wall Clock).
    2. Monitors System Resources (CPU, RAM).
    3. Monitors GPU Resources (VRAM, Utilization) using both PyTorch internals and NVML.
    4. Handles persistence of benchmark artifacts, respecting Hydra's directory context.
    """
    def __init__(self, model_name: str, dataset_name: str = "unknown", save_path: str = "benchmark_results.json", sampling_rate: float = 0.01):
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # --- Hydra Context Management ---
        # Hydra changes the CWD (Current Working Directory) to a date-stamped folder.
        # We generally want to save the summary JSON in the project root, not deep inside the Hydra run folder.
        try:
            orig_cwd = hydra.utils.get_original_cwd()
            self.save_path = os.path.join(orig_cwd, save_path)
        except Exception:
            # Fallback for non-Hydra executions
            self.save_path = save_path
            
        self.process = psutil.Process(os.getpid())
        self.sampling_rate = sampling_rate
        
        # State containers
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        self.monitoring_active = False

    def _monitor_gpu_usage(self):
        """
        Background daemon for polling GPU stats via NVML.
        Run in a separate thread to avoid blocking the training/inference loop.
        """
        while self.monitoring_active:
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # 1. Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    self.gpu_load_history.append(util)
                    if util > self.peak_gpu_util:
                        self.peak_gpu_util = util
                    
                    # 2. VRAM (Physical Memory Used)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_vram_mb = mem_info.used / (1024 ** 2)
                    if current_vram_mb > self.max_vram_system:
                        self.max_vram_system = current_vram_mb
                except Exception:
                    # Fail silently in thread to prevent crashing the benchmark
                    pass
            time.sleep(self.sampling_rate) 

    def start_measurement(self):
        """
        Prepares the environment and initiates background monitoring.
        """
        # 1. Garbage Collection (Reset State)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # 2. Reset Metrics
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        
        # 3. Baseline VRAM Check (NVML)
        if PYNVML_AVAILABLE:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.max_vram_system = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**2)
            except: pass

        # 4. Start Timers
        self.start_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        self.start_time = time.perf_counter()
        
        # 5. Launch Background Thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu_usage, daemon=True)
        self.monitor_thread.start()

    def end_measurement(self, task_name: str = "inference", extra_metrics: Optional[Dict[str, Any]] = None):
        """
        Stops monitoring, aggregates stats, and persists results.
        """
        # 1. Synchronization (Crucial for GPU timing)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 2. Stop Thread
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        end_time = time.perf_counter()
        duration = end_time - self.start_time

        # --- VRAM Metric Resolution ---
        vram_peak_torch = 0
        final_vram_metric = 0
        
        if torch.cuda.is_available():
            # torch.max_memory_allocated(): Precise memory actually used by tensors.
            # Ignores overhead/fragmentation/other processes.
            vram_peak_torch = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        # Decision Logic:
        # If PyTorch was used (vram_peak_torch > 0), it is the most accurate metric for the model.
        # If not (e.g., XGBoost on CPU or GPU via obscure library), fall back to System VRAM peak (NVML).
        if vram_peak_torch > 0:
            final_vram_metric = vram_peak_torch
        else:
            final_vram_metric = self.max_vram_system

        # --- CPU & RAM Aggregation ---
        # Include child processes (useful for DataLoaders workers)
        current_process = self.process
        all_procs = [current_process] + current_process.children(recursive=True)
        
        total_cpu_time_end = 0
        total_rss_mem = 0
        
        for p in all_procs:
            try:
                total_cpu_time_end += (p.cpu_times().user + p.cpu_times().system)
                total_rss_mem += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass 

        # Calculate Utilization %
        cpu_usage_precise = ((total_cpu_time_end - self.start_cpu_time) / duration) * 100
        ram_usage = total_rss_mem / (1024 ** 2) 

        avg_gpu_util = np.mean(self.gpu_load_history) if self.gpu_load_history else 0

        # --- Data Packaging ---
        data = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "task": task_name,
            "time_sec": round(duration, 4),
            "ram_mb": round(ram_usage, 2),
            
            # The "Official" VRAM metric for the report
            "vram_mb": round(final_vram_metric, 2),
            
            # Debugging metrics (kept for deep-dive analysis)
            "vram_system_peak_mb": round(self.max_vram_system, 2),
            "vram_torch_peak_mb": round(vram_peak_torch, 2),
            
            "cpu_percent_avg": round(cpu_usage_precise, 1),
            "gpu_util_percent_avg": round(avg_gpu_util, 1),
            "gpu_util_percent_peak": round(self.peak_gpu_util, 1)
        }
        
        if extra_metrics:
            data.update(extra_metrics)
        
        logger.info(f"--- Benchmark Results: {self.model_name} | Dataset: {self.dataset_name} | Task: {task_name} ---")
        self._save_to_file(data)
        return data

    def _save_to_file(self, data):
        """
        Appends results to the JSON ledger.
        Note: Not fully process-safe for massive parallel writes, but sufficient for sequential Hydra jobs.
        """
        history = []
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                pass
        
        history.append(data)
        
        # Atomic write simulation (write then close)
        with open(self.save_path, 'w') as f:
            json.dump(history, f, indent=4)