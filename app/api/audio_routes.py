# app/api/audio_routes.py (Updated for Celery)
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import os

from ..tasks.audio_tasks import process_audio_task, process_audio_quick_task
from ..celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["audio"])

# Response Models (keep your existing ones)
class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_time: str
    status_endpoint: str

@router.post("/process", response_model=TaskResponse)
async def process_audio_complete(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    include_translation: bool = Form(True),
    include_insights: bool = Form(True),
    background: bool = Form(True)
):
    """
    Complete audio-to-insights pipeline with Celery
    """
    
    # Validate audio file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    allowed_formats = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"]
    file_extension = os.path.splitext(audio.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_extension}. Supported: {allowed_formats}"
        )
    
    max_size = 100 * 1024 * 1024  # 100MB
    audio_bytes = await audio.read()
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(audio_bytes)/1024/1024:.1f}MB. Max: {max_size/1024/1024}MB"
        )
    
    try:
        if background:
            # Submit to Celery
            task = process_audio_task.delay(
                audio_bytes=audio_bytes,
                filename=audio.filename,
                language=language,
                include_translation=include_translation,
                include_insights=include_insights
            )
            
            logger.info(f"🎙️ Submitted audio processing task {task.id} for {audio.filename}")
            
            return TaskResponse(
                task_id=task.id,
                status="queued",
                message="Audio processing started. Check status at /audio/task/{task_id}",
                estimated_time="15-45 seconds",
                status_endpoint=f"/audio/task/{task.id}"
            )
        else:
            # Synchronous processing
            result = process_audio_task(
                audio_bytes=audio_bytes,
                filename=audio.filename,
                language=language,
                include_translation=include_translation,
                include_insights=include_insights
            )
            return result
            
    except Exception as e:
        logger.error(f"❌ Audio processing failed for {audio.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )

@router.post("/analyze", response_model=TaskResponse)
async def quick_audio_analysis(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    background: bool = Form(True)
):
    """
    Quick audio analysis with Celery
    """
    
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    allowed_formats = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"]
    file_extension = os.path.splitext(audio.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {file_extension}")
    
    audio_bytes = await audio.read()
    max_size = 50 * 1024 * 1024  # 50MB
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large for quick analysis: {len(audio_bytes)/1024/1024:.1f}MB. Max: 50MB"
        )
    
    try:
        if background:
            task = process_audio_quick_task.delay(
                audio_bytes=audio_bytes,
                filename=audio.filename,
                language=language
            )
            
            return TaskResponse(
                task_id=task.id,
                status="queued",
                message="Quick analysis started",
                estimated_time="10-20 seconds",
                status_endpoint=f"/audio/task/{task.id}"
            )
        else:
            result = process_audio_quick_task(
                audio_bytes=audio_bytes,
                filename=audio.filename,
                language=language
            )
            return result
            
    except Exception as e:
        logger.error(f"❌ Quick analysis failed for {audio.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get Celery task status with detailed progress"""
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == "PENDING":
            response = {
                "task_id": task_id,
                "status": "queued",
                "progress": 0,
                "message": "Task is waiting to be processed"
            }
        elif task.state == "PROCESSING":
            response = {
                "task_id": task_id,
                "status": "processing",
                "progress": task.info.get("progress", 0),
                "current_step": task.info.get("step", "unknown"),
                "filename": task.info.get("filename", "unknown")
            }
        elif task.state == "SUCCESS":
            response = {
                "task_id": task_id,
                "status": "completed",
                "progress": 100,
                "result": task.result,
                "processing_time": task.result.get("processing_time", 0) if task.result else 0
            }
        elif task.state == "FAILURE":
            response = {
                "task_id": task_id,
                "status": "failed",
                "error": str(task.info),
                "filename": task.info.get("filename", "unknown") if isinstance(task.info, dict) else "unknown"
            }
        else:
            response = {
                "task_id": task_id,
                "status": task.state.lower(),
                "info": task.info
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving task status")

@router.get("/tasks/active")
async def get_active_tasks():
    """Get all active audio processing tasks"""
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        
        if not active_tasks:
            return {"active_tasks": [], "total_active": 0}
        
        # Flatten active tasks from all workers
        all_active = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                if task['name'] in ['process_audio_task', 'process_audio_quick_task']:
                    all_active.append({
                        "task_id": task['id'],
                        "task_name": task['name'],
                        "worker": worker,
                        "args": task.get('args', [])
                    })
        
        return {
            "active_tasks": all_active,
            "total_active": len(all_active)
        }
        
    except Exception as e:
        logger.error(f"Error getting active tasks: {e}")
        return {"active_tasks": [], "total_active": 0, "error": str(e)}

@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a Celery task"""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return {"message": f"Task {task_id} cancelled successfully"}
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Error cancelling task")

@router.get("/queue/status")
async def get_queue_status():
    """Get overall queue status from Celery"""
    try:
        inspect = celery_app.control.inspect()
        
        # Get stats from all workers
        stats = inspect.stats()
        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()
        
        if not stats:
            return {
                "status": "no_workers",
                "message": "No Celery workers are running",
                "workers": 0
            }
        
        total_active = sum(len(tasks) for tasks in active.values()) if active else 0
        total_scheduled = sum(len(tasks) for tasks in scheduled.values()) if scheduled else 0
        total_reserved = sum(len(tasks) for tasks in reserved.values()) if reserved else 0
        
        worker_info = []
        for worker_name, worker_stats in stats.items():
            worker_info.append({
                "name": worker_name,
                "status": "online",
                "total_tasks": worker_stats.get("total", {}),
                "current_load": len(active.get(worker_name, [])) if active else 0
            })
        
        return {
            "status": "healthy",
            "workers": len(stats),
            "worker_info": worker_info,
            "queue_stats": {
                "active_tasks": total_active,
                "scheduled_tasks": total_scheduled,
                "reserved_tasks": total_reserved,
                "total_pending": total_active + total_scheduled + total_reserved
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "workers": 0
        }