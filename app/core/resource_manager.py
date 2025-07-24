import asyncio
import logging
import psutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time
import sys

logger = logging.getLogger(__name__)

# Optional torch import - gracefully handle if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU monitoring disabled")

class GPUResourceManager:
    """Manages GPU access using semaphores to prevent resource conflicts"""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.gpu_semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests: Dict[str, Dict] = {}
        self.request_counter = 0
        self.start_time = datetime.now()
        
        logger.info(f"GPU Resource Manager initialized with max_concurrent={max_concurrent}")
        
        if not TORCH_AVAILABLE:
            logger.info("Running in CPU-only mode (PyTorch not available)")
    
    async def acquire_gpu(self, request_id: str) -> bool:
        """Acquire GPU access for a request"""
        logger.info(f"Request {request_id} requesting GPU access")
        
        try:
            # Record request start
            self.active_requests[request_id] = {
                "start_time": datetime.now(),
                "status": "waiting"
            }
            
            # Wait for GPU access
            await self.gpu_semaphore.acquire()
            
            # Update status
            self.active_requests[request_id]["status"] = "processing"
            self.active_requests[request_id]["gpu_acquired_time"] = datetime.now()
            
            logger.info(f"Request {request_id} acquired GPU access")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acquire GPU for request {request_id}: {e}")
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            return False
    
    def release_gpu(self, request_id: str):
        """Release GPU access for a request"""
        try:
            self.gpu_semaphore.release()
            
            if request_id in self.active_requests:
                processing_time = datetime.now() - self.active_requests[request_id]["gpu_acquired_time"]
                logger.info(f"Request {request_id} released GPU after {processing_time.total_seconds():.2f}s")
                del self.active_requests[request_id]
            
        except Exception as e:
            logger.error(f"Error releasing GPU for request {request_id}: {e}")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU status"""
        gpu_info = {
            "torch_available": TORCH_AVAILABLE,
            "gpu_available": False,
            "gpu_count": 0,
            "active_requests": len(self.active_requests),
            "max_concurrent": self.max_concurrent,
            "requests_in_queue": max(0, len(self.active_requests) - self.max_concurrent)
        }
        
        if TORCH_AVAILABLE:
            try:
                gpu_info.update({
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count(),
                })
                
                if torch.cuda.is_available():
                    gpu_info.update({
                        "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                        "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                        "gpu_memory_cached": torch.cuda.memory_reserved(0),
                        "gpu_name": torch.cuda.get_device_name(0)
                    })
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")
                gpu_info["gpu_error"] = str(e)
        else:
            gpu_info["gpu_note"] = "PyTorch not installed - GPU monitoring unavailable"
        
        return gpu_info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform
        }

# Global resource manager instance
resource_manager = GPUResourceManager(max_concurrent=1)