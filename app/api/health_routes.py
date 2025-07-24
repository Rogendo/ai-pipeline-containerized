from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime

from core.resource_manager import resource_manager
from core.request_queue import request_queue
from models.model_loader import model_loader
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def basic_health():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version,
        "site_id": settings.site_id
    }

@router.get("/detailed")
async def detailed_health():
    """Detailed health check with full system status"""
    try:
        # Get all system information
        gpu_info = resource_manager.get_gpu_info()
        system_info = resource_manager.get_system_info()
        queue_status = request_queue.get_queue_status()
        model_status = model_loader.get_model_status()
        
        # Determine overall health
        overall_status = "healthy"
        issues = []
        
        # Check for issues
        if system_info["memory_percent"] > settings.alert_memory_usage:
            issues.append(f"High memory usage: {system_info['memory_percent']:.1f}%")
            overall_status = "degraded"
        
        if queue_status["queue_size"] > settings.alert_queue_size:
            issues.append(f"Queue nearly full: {queue_status['queue_size']}/{queue_status['max_queue_size']}")
            overall_status = "degraded"
        
        ready_models = model_loader.get_ready_models()
        failed_models = model_loader.get_failed_models()
        
        if len(ready_models) == 0:
            issues.append("No models are ready")
            overall_status = "unhealthy"
        elif len(failed_models) > 0:
            issues.append(f"Some models failed to load: {failed_models}")
            if overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": settings.app_version,
            "site_id": settings.site_id,
            "issues": issues,
            "system": system_info,
            "gpu": gpu_info,
            "queue": queue_status,
            "models": {
                "total": len(model_status),
                "ready": len(ready_models),
                "failed": len(failed_models),
                "ready_models": ready_models,
                "failed_models": failed_models,
                "details": model_status
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/models")
async def models_health():
    """Get detailed model status"""
    model_status = model_loader.get_model_status()
    ready_models = model_loader.get_ready_models()
    failed_models = model_loader.get_failed_models()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(model_status),
            "ready": len(ready_models),
            "failed": len(failed_models)
        },
        "ready_models": ready_models,
        "failed_models": failed_models,
        "details": model_status
    }

@router.get("/resources")
async def resources_health():
    """Get detailed resource status"""
    gpu_info = resource_manager.get_gpu_info()
    system_info = resource_manager.get_system_info()
    queue_status = request_queue.get_queue_status()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_info,
        "system": system_info,
        "queue": queue_status
    }
