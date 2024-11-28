import psutil
import torch
import gc
import logging
from typing import Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MEMORY_THRESHOLD = 0.9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResourceMonitor:
    """Monitors and manages system resources for the application."""
    
    @staticmethod
    def check_memory():
        """Check and manage memory usage"""
        memory = psutil.virtual_memory()
        if memory.percent > MEMORY_THRESHOLD * 100:
            logger.warning(f"High memory usage: {memory.percent}%")
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    @staticmethod
    def check_gpu():
        """Monitor GPU memory usage if available"""
        if DEVICE.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            logger.info(
                f"GPU Memory - Allocated: {memory_allocated/1024**2:.2f}MB, "
                f"Reserved: {memory_reserved/1024**2:.2f}MB"
            )

    @staticmethod
    def check_resources() -> Tuple[bool, str]:
        """
        Check system resources before processing
        
        Returns:
            Tuple[bool, str]: (is_ok, error_message)
        """
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > MEMORY_THRESHOLD * 100:
                return False, f"System memory usage too high ({memory.percent}%)"

            # Check GPU memory if available
            if DEVICE.type == 'cuda':
                gpu_memory_used = (
                    torch.cuda.memory_allocated() / 
                    torch.cuda.max_memory_allocated() 
                    if torch.cuda.max_memory_allocated() > 0 else 0
                )
                if gpu_memory_used > 0.9:
                    return False, f"GPU memory usage too high ({gpu_memory_used*100:.1f}%)"

            return True, ""
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return False, "Resource check failed"

    @staticmethod
    def log_resource_usage():
        """Log comprehensive resource usage statistics"""
        try:
            # System memory stats
            memory = psutil.virtual_memory()
            logger.info(
                f"System Memory - Total: {memory.total/1024**3:.1f}GB, "
                f"Available: {memory.available/1024**3:.1f}GB, "
                f"Used: {memory.percent}%"
            )
            
            # CPU stats
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"CPU Usage: {cpu_percent}%")
            
            # GPU stats if available
            if DEVICE.type == 'cuda':
                ResourceMonitor.check_gpu()
                
            # Process specific stats
            process = psutil.Process()
            logger.info(
                f"Process Memory Usage: {process.memory_info().rss/1024**2:.1f}MB, "
                f"CPU Usage: {process.cpu_percent()}%"
            )
            
        except Exception as e:
            logger.error(f"Error logging resource usage: {str(e)}")