# app/core/celery_monitor.py
import threading
import logging
from typing import Dict, Any
from datetime import datetime
from celery.events import EventReceiver
from ..celery_app import celery_app

logger = logging.getLogger(__name__)

class CeleryEventMonitor:
    def __init__(self):
        self.active_tasks = {}
        self.worker_stats = {}
        self.monitoring_thread = None
        
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.info("Event monitoring already running")
            return
            
        def monitor_worker():
            try:
                logger.info("Starting Celery event monitoring...")
                
                def on_task_started(event):
                    self.active_tasks[event['uuid']] = {
                        'name': event['name'],
                        'worker': event['hostname'],
                        'started': datetime.fromtimestamp(event['timestamp']).isoformat(),
                        'args': event.get('args', [])
                    }
                    logger.info(f"Task started: {event['name']} ({event['uuid'][:8]})")
                
                def on_task_succeeded(event):
                    task_info = self.active_tasks.pop(event['uuid'], {})
                    logger.info(f"Task completed: {task_info.get('name', 'unknown')} ({event['uuid'][:8]})")
                
                def on_task_failed(event):
                    task_info = self.active_tasks.pop(event['uuid'], {})
                    logger.warning(f"Task failed: {task_info.get('name', 'unknown')} ({event['uuid'][:8]})")
                
                def on_worker_heartbeat(event):
                    self.worker_stats[event['hostname']] = {
                        'last_heartbeat': datetime.fromtimestamp(event['timestamp']).isoformat(),
                        'status': 'online'
                    }
                
                # Start monitoring
                with celery_app.connection() as connection:
                    receiver = EventReceiver(
                        connection,
                        handlers={
                            'task-started': on_task_started,
                            'task-succeeded': on_task_succeeded,
                            'task-failed': on_task_failed,
                            'worker-heartbeat': on_worker_heartbeat,
                        }
                    )
                    receiver.capture(limit=None, timeout=None, wakeup=True)
                
            except Exception as e:
                logger.error(f"Event monitoring failed: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitoring_thread.start()
        logger.info("Celery event monitoring thread started")
    
    def get_active_tasks(self) -> Dict[str, Any]:
        """Get currently active tasks"""
        return {
            "active_tasks": list(self.active_tasks.values()),
            "total_active": len(self.active_tasks),
            "data_source": "celery_events"
        }
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "workers": self.worker_stats,
            "total_workers": len(self.worker_stats)
        }

# Global instance
celery_monitor = CeleryEventMonitor()