from pydantic_settings import BaseSettings
from typing import Optional
import os

import redis

class Settings(BaseSettings):
    # Application
    app_name: str = "AI Pipeline"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    
    # Resource Management
    max_concurrent_gpu_requests: int = 1
    max_queue_size: int = 20
    request_timeout: int = 300
    queue_monitor_interval: int = 30
    
    # Model Configuration
    model_cache_size: int = 8192
    cleanup_interval: int = 3600
    enable_model_loading: bool = False
    
    # Security
    site_id: str = "unknown-site"
    data_retention_hours: int = 24
    
    # Performance
    enable_queue_metrics: bool = True
    alert_queue_size: int = 15
    alert_memory_usage: int = 90
    
    # Paths
    models_path: str = "/app/models"
    logs_path: str = "/app/logs"
    temp_path: str = "/app/temp"
    
    redis_url: str = "redis://localhost:6379/0"
    redis_task_db: int = 1 
    
    class Config:
        env_file = ".env"

settings = Settings()

redis_client = redis.from_url(settings.redis_url)
redis_task_client = redis.from_url(f"redis://localhost:6379/{settings.redis_task_db}")
