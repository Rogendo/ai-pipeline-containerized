from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import os

from ..core.audio_pipeline import audio_pipeline
from ..models.model_loader import model_loader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["audio"])

# Response Models
class AudioInfo(BaseModel):
    filename: str
    file_size_mb: float
    language_specified: Optional[str]
    processing_time: float

class ProcessingStep(BaseModel):
    duration: float
    status: str
    entities_found: Optional[int] = None
    confidence: Optional[float] = None
    output_length: Optional[int] = None
    summary_length: Optional[int] = None
    enabled: Optional[bool] = None
    error: Optional[str] = None

class CaseOverview(BaseModel):
    primary_language: str
    key_entities: Dict[str, int]
    case_complexity: str

class RiskAssessment(BaseModel):
    risk_indicators_found: int
    risk_level: str
    priority: str
    confidence: float

class KeyInformation(BaseModel):
    main_category: str
    sub_category: str
    intervention_needed: str
    summary: str

class EntitiesDetail(BaseModel):
    persons: List[str]
    locations: List[str]
    organizations: List[str]
    key_dates: List[str]

class Insights(BaseModel):
    case_overview: CaseOverview
    risk_assessment: RiskAssessment
    key_information: KeyInformation
    entities_detail: EntitiesDetail

class PipelineInfo(BaseModel):
    total_time: float
    models_used: List[str]
    timestamp: str

class CompleteAudioResponse(BaseModel):
    audio_info: AudioInfo
    transcript: str
    translation: Optional[str]
    entities: Dict[str, List[str]]
    classification: Dict[str, Any]
    summary: str
    insights: Optional[Insights]
    processing_steps: Dict[str, ProcessingStep]
    pipeline_info: PipelineInfo

class QuickAnalysisResponse(BaseModel):
    transcript: str
    summary: str
    main_category: str
    priority: str
    risk_level: str
    processing_time: float

@router.post("/process", response_model=CompleteAudioResponse)
async def process_audio_complete(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    include_translation: bool = Form(True),
    include_insights: bool = Form(True)
):
    """
    Complete audio-to-insights pipeline
    
    Processes audio through: Whisper → NER → Classification → Translation → Summarization → Insights
    
    Parameters:
    - audio: Audio file (wav, mp3, flac, m4a, ogg, webm)
    - language: Language code (e.g., 'sw', 'en') or auto-detect if None
    - include_translation: Whether to translate to English
    - include_insights: Whether to generate case insights
    """
    
    # Check pipeline readiness
    readiness = audio_pipeline.check_pipeline_readiness()
    if not readiness["pipeline_ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Audio pipeline not ready. Missing models: {readiness['missing_models']}"
        )
    
    # Validate audio file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Check file format
    allowed_formats = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"]
    file_extension = os.path.splitext(audio.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file_extension}. Supported: {allowed_formats}"
        )
    
    # Check file size (100MB limit)
    max_size = 100 * 1024 * 1024  # 100MB
    audio_bytes = await audio.read()
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(audio_bytes)/1024/1024:.1f}MB. Max: {max_size/1024/1024}MB"
        )
    
    try:
        logger.info(f"🎙️ Starting complete audio processing for {audio.filename}")
        
        # Process through complete pipeline
        result = await audio_pipeline.process_audio_complete(
            audio_bytes=audio_bytes,
            filename=audio.filename,
            language=language,
            include_translation=include_translation,
            include_insights=include_insights
        )
        
        logger.info(f"🎉 Complete audio processing finished for {audio.filename}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Complete audio processing failed for {audio.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )

@router.post("/analyze", response_model=QuickAnalysisResponse)
async def quick_audio_analysis(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """
    Quick audio analysis - essentials only
    
    Returns: transcript, summary, category, priority, risk level
    """
    
    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    allowed_formats = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"]
    file_extension = os.path.splitext(audio.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_extension}"
        )
    
    audio_bytes = await audio.read()
    max_size = 50 * 1024 * 1024  # 50MB limit for quick analysis
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large for quick analysis: {len(audio_bytes)/1024/1024:.1f}MB. Max: 50MB"
        )
    
    try:
        start_time = datetime.now()
        
        # Process with minimal pipeline
        result = await audio_pipeline.process_audio_complete(
            audio_bytes=audio_bytes,
            filename=audio.filename,
            language=language,
            include_translation=False,
            include_insights=False
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract essentials
        classification = result.get("classification", {})
        insights = result.get("insights", {})
        risk_assessment = insights.get("risk_assessment", {}) if insights else {}
        
        return QuickAnalysisResponse(
            transcript=result["transcript"],
            summary=result["summary"],
            main_category=classification.get("main_category", "unknown"),
            priority=classification.get("priority", "medium"),
            risk_level=risk_assessment.get("risk_level", "unknown"),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Quick analysis failed for {audio.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Quick analysis failed: {str(e)}"
        )

@router.get("/pipeline/status")
async def get_pipeline_status():
    """Get audio pipeline readiness status"""
    return audio_pipeline.check_pipeline_readiness()

@router.get("/pipeline/info")
async def get_pipeline_info():
    """Get detailed pipeline information"""
    readiness = audio_pipeline.check_pipeline_readiness()
    
    return {
        "pipeline_ready": readiness["pipeline_ready"],
        "models_status": readiness["models"],
        "missing_models": readiness["missing_models"],
        "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"],
        "max_file_size_mb": 100,
        "endpoints": {
            "complete_analysis": "/audio/process",
            "quick_analysis": "/audio/analyze",
            "pipeline_status": "/audio/pipeline/status"
        },
        "pipeline_flow": [
            "Audio Upload",
            "Whisper Transcription", 
            "Parallel NLP Analysis (NER + Classification + Translation + Summarization)",
            "Insights Generation",
            "Complete Response"
        ]
    }

@router.post("/demo")
async def audio_demo():
    """Demo endpoint with usage examples"""
    pipeline_status = audio_pipeline.check_pipeline_readiness()
    
    return {
        "demo_info": {
            "complete_analysis": {
                "endpoint": "/audio/process",
                "description": "Full audio-to-insights pipeline",
                "example": "curl -X POST -F 'audio=@case.wav' -F 'language=sw' -F 'include_translation=true' http://localhost:8000/audio/process"
            },
            "quick_analysis": {
                "endpoint": "/audio/analyze", 
                "description": "Essential insights only (faster)",
                "example": "curl -X POST -F 'audio=@case.wav' -F 'language=sw' http://localhost:8000/audio/analyze"
            }
        },
        "pipeline_status": pipeline_status,
        "expected_output": {
            "transcript": "Audio transcription in original language",
            "translation": "English translation (if enabled)",
            "entities": "People, places, organizations mentioned",
            "classification": "Case category, priority, confidence",
            "summary": "Concise case summary",
            "insights": "Risk assessment and key information"
        }
    }