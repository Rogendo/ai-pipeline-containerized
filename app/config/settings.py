from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

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
    enable_model_loading: bool = True
    
    # Security
    site_id: str = "unknown-site"
    data_retention_hours: int = 24
    
    # Performance
    enable_queue_metrics: bool = True
    alert_queue_size: int = 15
    alert_memory_usage: int = 90
    
    # Paths - Auto-detect environment
    models_path: str = "/app/models"
    logs_path: str = "/app/logs"
    temp_path: str = "/app/temp"
    
    # Redis Configuration - Updated for Docker
    redis_url: str = "redis://localhost:6379/0"
    redis_task_db: int = 1
    
    # Docker detection
    docker_container: bool = False
    
    def __post_init__(self):
        """Auto-detect if running in container or locally"""
        # Detect if running in Docker container
        if (os.path.exists("/.dockerenv") or 
            os.environ.get("DOCKER_CONTAINER") or 
            self.docker_container):
            # Running in container - use container paths
            self.models_path = "/app/models"
            # Update Redis URL for Docker network if not explicitly set
            if self.redis_url == "redis://localhost:6379/0":
                self.redis_url = "redis://redis:6379/0"  # Use Docker service name
        else:
            # Running locally - use relative paths
            project_root = Path(__file__).parent.parent.parent
            self.models_path = str(project_root / "models")
            self.logs_path = str(project_root / "logs")
            self.temp_path = str(project_root / "temp")
        
        # Create directories if they don't exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """Get absolute path for a model"""
        return os.path.join(self.models_path, model_name)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Initialize settings and auto-detect paths
settings = Settings()
settings.__post_init__()

# Redis clients with error handling
try:
    redis_client = redis.from_url(settings.redis_url)
    redis_task_client = redis.from_url(f"redis://{'redis' if settings.docker_container else 'localhost'}:6379/{settings.redis_task_db}")
    
    # Test connection
    redis_client.ping()
    print(f"✅ Redis connected: {settings.redis_url}")
    
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    redis_client = None
    redis_task_client = None