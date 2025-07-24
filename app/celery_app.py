# app/celery_app.py
from celery import Celery
import os

# Celery configuration
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "audio_pipeline",
    broker=broker_url,
    backend=result_backend,
    include=["app.tasks.audio_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Results
    result_expires=3600,  # 1 hour
    result_backend_max_retries=10,
    
    # Task execution
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Important for GPU tasks
    
    # Routing
    task_routes={
        'app.tasks.audio_tasks.process_audio_task': {'queue': 'audio_processing'},
        'app.tasks.audio_tasks.process_audio_quick_task': {'queue': 'audio_quick'},
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)