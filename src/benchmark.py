import time
import torch
import psutil
import os
import json
import gc
import threading
import numpy as np

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: 'pynvml' not installed. Hardware VRAM cannot be measured accurately.")

class PerformanceMonitor:
    def __init__(self, model_name, save_path="benchmark_results.json", sampling_rate=0.01):
        self.model_name = model_name
        self.save_path = save_path
        self.process = psutil.Process(os.getpid())
        self.sampling_rate = sampling_rate # Higher frequency (e.g., 10ms)
        
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        self.monitoring_active = False

    def _monitor_gpu_usage(self):
        """Background thread: Captures hardware metrics with high frequency."""
        while self.monitoring_active:
            if PYNVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # 1. GPU Utilization (%)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    self.gpu_load_history.append(util)
                    if util > self.peak_gpu_util:
                        self.peak_gpu_util = util
                    
                    # 2. Real Hardware VRAM Usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_vram_mb = mem_info.used / (1024 ** 2)
                    if current_vram_mb > self.max_vram_system:
                        self.max_vram_system = current_vram_mb
                        
                except Exception:
                    pass
            
            time.sleep(self.sampling_rate) 

    def start_measurement(self):
        # 1. Hardware Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # 2. Reset Statistics
        self.gpu_load_history = []
        self.max_vram_system = 0 
        self.peak_gpu_util = 0
        
        # Initialize VRAM base level
        if PYNVML_AVAILABLE:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.max_vram_system = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**2)
            except: pass

        # 3. Precise CPU timing (using Process Time instead of percent)
        self.start_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        self.start_time = time.perf_counter() # Use high-resolution timer
        
        # 4. Start Background Thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu_usage)
        self.monitor_thread.start()

    def end_measurement(self, task_name="inference", extra_metrics=None):
        # 1. Sync GPU and Stop Monitor
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.monitoring_active = False
        self.monitor_thread.join()
        
        end_time = time.perf_counter()
        end_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        
        # 2. Calculate Metrics
        duration = end_time - self.start_time
        
        # CPU usage calculation: (CPU_Time_Used / Wall_Clock_Time) * 100
        # This gives the average across all cores (can be > 100%)
        cpu_usage_precise = ((end_cpu_time - self.start_cpu_time) / duration) * 100 if duration > 0 else 0
        
        ram_usage = self.process.memory_info().rss / (1024 ** 2)
        
        # PyTorch specific VRAM
        vram_peak_torch = 0
        if torch.cuda.is_available():
            vram_peak_torch = torch.cuda.max_memory_allocated() / (1024 ** 2)

        avg_gpu_util = np.mean(self.gpu_load_history) if self.gpu_load_history else 0

        # 3. Data Structure
        data = {
            "model": self.model_name,
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
        
        print(f"--- Results {self.model_name} ({task_name}) ---")
        print(f"Time: {data['time_sec']}s | CPU: {data['cpu_percent_avg']}%")
        print(f"GPU Peak: {data['gpu_util_percent_peak']}% | VRAM Peak: {data['vram_system_peak_mb']} MB")
        
        self._save_to_file(data)
        return data

    def _save_to_file(self, data):
        # Implementation remains similar but with more robust error handling
        history = []
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                try: history = json.load(f)
                except: pass
        history.append(data)
        with open(self.save_path, 'w') as f:
            json.dump(history, f, indent=4)