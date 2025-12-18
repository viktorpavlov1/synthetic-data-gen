
import time
import threading
import subprocess
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitors GPU power usage using nvidia-smi in a background thread."""
    
    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.power_readings = []
        self.running = False
        self.thread = None
        
    def start(self):
        """Start monitoring."""
        self.running = True
        self.power_readings = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring and return readings."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        return self.power_readings
        
    def _monitor_loop(self):
        """Loop to query nvidia-smi."""
        while self.running:
            try:
                # Run nvidia-smi query
                # Format: power.draw (in Watts)
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    # Get output, could be multiple lines if multiple GPUs, take first for now or sum?
                    # Assuming single GPU or interested in total
                    output = result.stdout.strip()
                    if output:
                        # Sum all GPUs if multiple lines
                        try:
                            watts = sum(float(x) for x in output.split('\n') if x.strip())
                            self.power_readings.append(watts)
                        except ValueError:
                            pass
            except Exception as e:
                logger.debug(f"Error querying GPU power: {e}")
                
            time.sleep(self.sample_interval)

    def get_avg_power(self) -> float:
        """Get average power in Watts."""
        if not self.power_readings:
            return 0.0
        return float(np.mean(self.power_readings))
    
    def get_max_power(self) -> float:
        if not self.power_readings:
            return 0.0
        return float(np.max(self.power_readings))


class PerformanceTracker:
    """Tracks timing and energy metrics for generation runs."""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.start_time = 0.0
        self.end_time = 0.0
        self.image_times = [] # List of durations per batch or image if accurate
        
    def start_tracking(self):
        """Start tracking time and power."""
        self.start_time = time.time()
        self.gpu_monitor.start()
        
    def stop_tracking(self):
        """Stop tracking."""
        self.end_time = time.time()
        self.gpu_monitor.stop()
        
    def get_metrics(self, num_images: int) -> Dict[str, float]:
        """
        Calculate final metrics.
        
        Args:
            num_images: Number of images generated
            
        Returns:
            Dictionary with metrics:
            - total_time_seconds
            - seconds_per_image
            - avg_power_watts
            - energy_consumed_wh
            - energy_per_image_wh
        """
        total_time = self.end_time - self.start_time
        avg_power = self.gpu_monitor.get_avg_power()
        
        # Energy (Wh) = Power (W) * Time (h)
        # Time in hours = total_time / 3600
        energy_wh = avg_power * (total_time / 3600.0)
        
        metrics = {
            "total_time_seconds": total_time,
            "seconds_per_image": total_time / max(1, num_images),
            "avg_power_watts": avg_power,
            "peak_power_watts": self.gpu_monitor.get_max_power(),
            "total_energy_wh": energy_wh,
            "energy_per_image_wh": energy_wh / max(1, num_images)
        }
        
        return metrics
