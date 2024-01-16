import threading
import time
import GPUtil
import psutil
import cpuinfo
import tensorflow as tf
import numpy as np

class PerformanceMonitor(threading.Thread):
    def __init__(self, interval):
        threading.Thread.__init__(self)

        # CPU Information
        cpu_info = cpuinfo.get_cpu_info()
        self.cpu_model = cpu_info['brand_raw']

        # RAM Information
        ram_info = psutil.virtual_memory()
        self.total_ram = ram_info.total / (1024 ** 3)  # Convert to GB

        # GPU Information
        tf_gpus = tf.config.list_physical_devices('GPU')
        self.tf_gpu_info = [str(gpu) for gpu in tf_gpus]

        self.gpu_info = None
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            self.gpu_info = {
                'GPU ID': gpu.id,
                'GPU Name': gpu.name,
                'Total Memory (MB)': gpu.memoryTotal,
                'Driver': gpu.driver,
            }

        self.interval = interval
        self.running = True
        self.cpu_usage_samples = []
        self.ram_usage_samples = []

        self.cpu_usage_samples_percent = []
        self.gpu_load_samples_percent = []
        self.ram_usage_samples_percent = []

        self.gpu_memory_used = []
        self.gpu_temperature = []

    def run(self):
        while self.running:
            # Sample CPU, GPU, and RAM usage
            self.cpu_usage_samples_percent.append(psutil.cpu_percent())
            self.cpu_usage_samples.append(psutil.cpu_stats())
            self.ram_usage_samples_percent.append(psutil.virtual_memory().percent)
            self.ram_usage_samples.append(psutil.virtual_memory())
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_load_samples_percent.append(gpus[0].load * 100)
                self.gpu_memory_used.append(gpus[0].temperature)
                self.gpu_temperature.append(gpus[0].memoryUsed)
            time.sleep(self.interval)

    def get_percentiles(self, samples, percentiles):
        return {p: np.percentile(samples, p) for p in percentiles}

    def get_statistics(self):
        stats = {}
        stats["CPU Model"] = self.cpu_model
        if self.cpu_usage_samples_percent:
            stats['CPU Usage (%)'] = self.get_percentiles(self.cpu_usage_samples_percent, [50, 95, 99])
        if self.cpu_usage_samples:
            stats['CPU Usage'] = self.cpu_usage_samples
        stats["TF GPU"] = self.tf_gpu_info
        if self.gpu_info:
            stats.update(self.gpu_info)
        if self.gpu_load_samples_percent:
            stats['GPU Load'] = self.get_percentiles(self.gpu_load_samples_percent, [50, 95, 99])
        if self.gpu_load_samples_percent:
            stats['GPU Memory Used (MB)'] = self.gpu_memory_used
        if self.gpu_load_samples_percent:
            stats['GPU Temperature (C)'] = self.gpu_temperature
        stats["Total RAM"] = self.total_ram
        if self.ram_usage_samples_percent:
            stats['RAM Usage (%)'] = self.get_percentiles(self.ram_usage_samples_percent, [50, 95, 99])
        if self.ram_usage_samples:
            stats['RAM Usage'] = self.ram_usage_samples
        return stats

    def stop(self):
        self.running = False
