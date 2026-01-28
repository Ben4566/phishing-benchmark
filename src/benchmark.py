import time
import torch
import psutil
import os
import json
import gc
import threading
import numpy as np
from typing import Optional, Dict, Any

# --- HARDWARE TELEMETRY SETUP ---
# We utilize the NVIDIA Management Library (NVML) for high-precision, 
# system-level GPU monitoring (VRAM, Utilization) beyond what PyTorch reports.
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: 'pynvml' bindings not found. High-precision VRAM monitoring disabled.")

class PerformanceMonitor:
    """
    Orchestrates hardware-aware benchmarking.
    
    Features:
    - Asynchronous Resource Polling: Uses a background thread to sample GPU metrics 
      without blocking the main training/inference loop.
    - Precise Timing: Handles CUDA synchronization to ensure accurate duration measurement
      of asynchronous GPU kernels.
    - Full-Stack Observability: Tracks CPU, System RAM, GPU Utilization, and VRAM (Torch vs. System).
    """
    
    def __init__(self, model_name: str, dataset_name: str = "unknown", save_path: str = "benchmark_results.json", sampling_rate: float = 0.01):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.process = psutil.Process(os.getpid())
        self.sampling_rate = sampling_rate # 10ms sampling interval for high resolution
        
        # Telemetry storage
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        self.monitoring_active = False

    def _monitor_gpu_usage(self):
        """
        Background Thread: Continuously polls hardware counters.
        Running this in a separate thread ensures we capture transient load spikes 
        that occur during the model's forward/backward passes.
        """
        while self.monitoring_active:
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # 1. GPU Compute Utilization (%)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    self.gpu_load_history.append(util)
                    if util > self.peak_gpu_util:
                        self.peak_gpu_util = util
                    
                    # 2. System-Level VRAM Usage (includes CUDA Context + Driver overhead)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_vram_mb = mem_info.used / (1024 ** 2)
                    if current_vram_mb > self.max_vram_system:
                        self.max_vram_system = current_vram_mb
                except Exception:
                    pass
            # Yield control to prevent GIL contention
            time.sleep(self.sampling_rate) 

    def start_measurement(self):
        """
        Prepares the environment for a 'cold start' measurement to ensure consistency.
        """
        # 1. Force Garbage Collection to clear Python objects
        gc.collect()
        
        # 2. Clear PyTorch CUDA Cache to reset internal memory allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # IMPORTANT: Synchronize to ensure all previous GPU ops are done before timer starts
            torch.cuda.synchronize()
        
        # Reset metrics
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        
        # Baseline VRAM check
        if PYNVML_AVAILABLE:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.max_vram_system = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**2)
            except: pass

        # 3. Snapshot CPU times (User + System)
        self.start_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        self.start_time = time.perf_counter()
        
        # 4. Launch telemetry thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu_usage)
        self.monitor_thread.start()

    def end_measurement(self, task_name: str = "inference", extra_metrics: Optional[Dict[str, Any]] = None):
        """
        Concludes measurement, ensuring all async GPU work is finalized.
        """
        # 1. CUDA Synchronization: Wait for GPU to finish all kernels.
        # Without this, we would only measure the time to launch kernels, not execute them.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Stop telemetry
        self.monitoring_active = False
        self.monitor_thread.join()
        
        # 2. Calculate Durations & Resources
        end_time = time.perf_counter()
        end_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        
        duration = end_time - self.start_time
        # CPU Usage relative to the specific process execution time
        cpu_usage_precise = ((end_cpu_time - self.start_cpu_time) / duration) * 100 if duration > 0 else 0
        ram_usage = self.process.memory_info().rss / (1024 ** 2)
        
        # PyTorch-specific VRAM (Tensor data only, excludes context)
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
            "vram_system_peak_mb": round(self.max_vram_system, 2), # Driver perspective
            "vram_torch_peak_mb": round(vram_peak_torch, 2),     # PyTorch perspective
            "cpu_percent_avg": round(cpu_usage_precise, 1),
            "gpu_util_percent_avg": round(avg_gpu_util, 1),
            "gpu_util_percent_peak": round(self.peak_gpu_util, 1)
        }
        
        if extra_metrics:
            data.update(extra_metrics)
        
        print(f"--- Results {self.model_name} | Data: {self.dataset_name} ({task_name}) ---")
        print(f"Time: {data['time_sec']}s | CPU: {data['cpu_percent_avg']}%")
        print(f"GPU Peak: {data['gpu_util_percent_peak']}% | VRAM Peak: {data['vram_system_peak_mb']} MB")
        
        self._save_to_file(data)
        return data

    def _save_to_file(self, data):
        history = []
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                try: history = json.load(f)
                except: pass
        history.append(data)
        with open(self.save_path, 'w') as f:
            json.dump(history, f, indent=4)