from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from datetime import datetime

from ..models.qa_model import qa_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/qa", tags=["quality_assurance"])

class QARequest(BaseModel):
    transcript: str
    threshold: float = None
    return_raw: bool = False

class SubmetricResult(BaseModel):
    submetric: str
    prediction: bool
    score: str
    probability: float = None

class QAResponse(BaseModel):
    processing_time: float
    model_info: dict
    timestamp: str
    evaluations: dict

@router.post("/evaluate", response_model=QAResponse)
async def evaluate_transcript(request: QARequest):
    """Evaluate call center transcript against QA metrics"""
    
    # Check if QA model is loaded
    if not qa_model.is_ready():
        raise HTTPException(
            status_code=503, 
            detail="QA model not ready. Check /health/models for status."
        )
    
    if not request.transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="Transcript cannot be empty"
        )
    
    try:
        start_time = datetime.now()
        
        # Evaluate transcript
        evaluation = qa_model.predict(
            request.transcript,
            threshold=request.threshold,
            return_raw=request.return_raw
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get model info
        model_info = qa_model.get_model_info()
        
        logger.info(f"QA evaluation processed {len(request.transcript)} chars in {processing_time:.3f}s")
        
        return QAResponse(
            evaluations=evaluation,
            processing_time=processing_time,
            model_info=model_info,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"QA evaluation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

@router.get("/info")
async def get_qa_info():
    """Get QA model information"""
    if not qa_model.is_ready():
        return {
            "status": "not_ready",
            "message": "QA model not loaded"
        }

    return {
        "status": "ready",
        "model_info": qa_model.get_model_info()
    }

@router.post("/demo")
async def qa_demo():
    """Demo endpoint with sample transcript"""
    demo_transcript = (
        "Agent: Good morning! Thank you for calling TechSupport. My name is Alex. How can I help you today? "
        "Customer: Hi, I'm having issues with my internet connection. "
        "Agent: I'm sorry to hear that. Let me help you with that. Could you please tell me what exactly is happening? "
        "Customer: It keeps disconnecting every few minutes. "
        "Agent: I understand how frustrating that must be. Let me check your connection settings. "
        "Could you please hold for a moment while I investigate this? [Places on hold] "
        "Agent: Thank you for holding. I've found the issue - it's a configuration problem. "
        "I'll guide you through the steps to fix it. First, please open your network settings..."
    )
    
    request = QARequest(transcript=demo_transcript)
    return await evaluate_transcript(request)