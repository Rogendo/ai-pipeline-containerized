import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config.settings import settings
from .api import health_routes, queue_routes, ner_routes, translator_routes, summarizer_routes
from .models.model_loader import model_loader
from .core.resource_manager import resource_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Site ID: {settings.site_id}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize models if enabled
    if settings.enable_model_loading:
        logger.info("Model loading enabled - starting model initialization...")
        await model_loader.load_all_models()
    else:
        logger.info("Model loading disabled - models will show as 'not ready'")
    
    logger.info("Application startup complete")
    
    yield
    
    logger.info("Application shutdown")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Container-Baked AI Pipeline for Multi-Modal Processing",
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_routes.router)
app.include_router(queue_routes.router)
app.include_router(ner_routes.router)
app.include_router(translator_routes.router)
app.include_router(summarizer_routes.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "site_id": settings.site_id,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "detailed_health": "/health/detailed", 
            "models": "/health/models",
            "resources": "/health/resources",
            "queue": "/queue/status"
        }
    }

@app.get("/info")
async def app_info():
    """Application information"""
    gpu_info = resource_manager.get_gpu_info()
    system_info = resource_manager.get_system_info()
    
    return {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "site_id": settings.site_id,
            "debug": settings.debug
        },
        "configuration": {
            "max_concurrent_gpu_requests": settings.max_concurrent_gpu_requests,
            "max_queue_size": settings.max_queue_size,
            "request_timeout": settings.request_timeout,
            "model_loading_enabled": settings.enable_model_loading
        },
        "system": system_info,
        "gpu": gpu_info
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
