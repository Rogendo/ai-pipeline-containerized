# app/tasks/audio_tasks.py (Updated)
import json
import socket
from celery import current_task
from celery.signals import worker_init
from ..celery_app import celery_app
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from ..config.settings import redis_task_client


logger = logging.getLogger(__name__)

# Global model loader for Celery worker
worker_model_loader = None

@worker_init.connect
def init_worker(**kwargs):
    """Initialize models when Celery worker starts"""
    global worker_model_loader
    
    logger.info("🔄 Initializing models in Celery worker...")
    
    try:
        # Import here to avoid circular imports
        from ..models.model_loader import ModelLoader
        
        # Create a new model loader instance for this worker
        worker_model_loader = ModelLoader()
        
        # Load models synchronously (Celery workers can't use async)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(worker_model_loader.load_all_models())
        loop.close()
        
        logger.info("✅ Models loaded successfully in Celery worker")
        
        # Log which models are ready
        ready_models = worker_model_loader.get_ready_models()
        logger.info(f"✅ Ready models in worker: {ready_models}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models in Celery worker: {e}")
        worker_model_loader = None

def get_worker_models():
    """Get the worker's model loader instance"""
    global worker_model_loader
    return worker_model_loader

@celery_app.task(bind=True, name="process_audio_task")
def process_audio_task(
    self,
    audio_bytes: bytes,
    filename: str,
    language: Optional[str] = None,
    include_translation: bool = True,
    include_insights: bool = True
):
    """
    Celery task for complete audio processing with worker models
    """
    # Store in Redis for reliable tracking
    task_info = {
        "task_id": self.request.id,
        "filename": filename,
        "started": datetime.now().isoformat(),
        "worker": socket.gethostname(),
        "status": "processing"
    }
    
    redis_task_client.hset(
        "active_audio_tasks", 
        self.request.id, 
        json.dumps(task_info)
    )
    try:
        # Update task state
        self.update_state(
            state="PROCESSING",
            meta={
                "step": "initializing",
                "filename": filename,
                "progress": 0
            }
        )
        
        # Get worker models
        models = get_worker_models()
        if not models:
            raise RuntimeError("Models not loaded in Celery worker. Check worker logs.")
        
        # Check pipeline readiness using worker models
        required_models = ["whisper", "ner", "classifier_model", "translator", "summarizer"]
        missing_models = []
        for model_name in required_models:
            if not models.is_model_ready(model_name):
                missing_models.append(model_name)
        
        if missing_models:
            raise RuntimeError(f"Pipeline not ready in worker. Missing models: {missing_models}")
        
        # Process the audio using worker models
        result = _process_audio_sync_worker(
            self, models, audio_bytes, filename, language, 
            include_translation, include_insights
        )
        
        # Remove from Redis on completion
        redis_task_client.hdel("active_audio_tasks", self.request.id)
        
        return {
            "status": "completed",
            "filename": filename,
            "processing_time": result["pipeline_info"]["total_time"],
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Audio processing task failed: {e}")
        # Better exception handling for Celery
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "filename": filename,
                "error_type": type(e).__name__
            }
        )
        # Re-raise with proper exception type
        raise type(e)(str(e))

def _process_audio_sync_worker(
    task_instance,
    models,  # Use worker models instead of global model_loader
    audio_bytes: bytes,
    filename: str,
    language: Optional[str],
    include_translation: bool,
    include_insights: bool
) -> Dict[str, Any]:
    """
    Synchronous audio processing using worker models
    """
    start_time = datetime.now()
    processing_steps = {}
    
    # Step 1: Transcription
    task_instance.update_state(
        state="PROCESSING",
        meta={"step": "transcription", "progress": 10}
    )
    
    step_start = datetime.now()
    whisper_model = models.models.get("whisper")
    if not whisper_model:
        raise RuntimeError("Whisper model not available in worker")
        
    transcript = whisper_model.transcribe_audio_bytes(audio_bytes, language=language)
    
    processing_steps["transcription"] = {
        "duration": (datetime.now() - step_start).total_seconds(),
        "status": "completed",
        "output_length": len(transcript)
    }
    
    # Step 2: Translation (if enabled)
    translation = None
    if include_translation:
        task_instance.update_state(
            state="PROCESSING",
            meta={"step": "translation", "progress": 30}
        )
        
        step_start = datetime.now()
        try:
            translator_model = models.models.get("translator")
            if translator_model:
                translation = translator_model.translate(transcript)
                
                processing_steps["translation"] = {
                    "duration": (datetime.now() - step_start).total_seconds(),
                    "status": "completed",
                    "output_length": len(translation)
                }
            else:
                raise RuntimeError("Translator model not available")
        except Exception as e:
            processing_steps["translation"] = {
                "duration": (datetime.now() - step_start).total_seconds(),
                "status": "failed",
                "error": str(e)
            }
            translation = None
    
    # Step 3: NLP Processing
    task_instance.update_state(
        state="PROCESSING",
        meta={"step": "nlp_analysis", "progress": 50}
    )
    
    nlp_text = translation if translation else transcript
    nlp_source = "translated_text" if translation else "original_transcript"
    
    # NER
    step_start = datetime.now()
    try:
        ner_model = models.models.get("ner")
        if not ner_model:
            raise RuntimeError("NER model not available")
        entities = ner_model.extract_entities(nlp_text, flat=False)
        ner_status = {
            "result": entities,
            "duration": (datetime.now() - step_start).total_seconds(),
            "status": "completed"
        }
    except Exception as e:
        ner_status = {
            "result": {},
            "duration": (datetime.now() - step_start).total_seconds(),
            "status": "failed",
            "error": str(e)
        }
    
    # Classification
    task_instance.update_state(
        state="PROCESSING",
        meta={"step": "classification", "progress": 70}
    )
    
    step_start = datetime.now()
    try:
        classifier_model = models.models.get("classifier_model")
        if not classifier_model:
            raise RuntimeError("Classifier model not available")
        classification = classifier_model.classify(nlp_text)
        classifier_status = {
            "result": classification,
            "duration": (datetime.now() - step_start).total_seconds(),
            "status": "completed"
        }
    except Exception as e:
        classifier_status = {
            "result": {},
            "duration": (datetime.now() - step_start).total_seconds(),
            "status": "failed",
            "error": str(e)
        }
    
    # Summarization
    task_instance.update_state(
        state="PROCESSING",
        meta={"step": "summarization", "progress": 85}
    )
    
    step_start = datetime.now()
    try:
        summarizer_model = models.models.get("summarizer")
        if not summarizer_model:
            raise RuntimeError("Summarizer model not available")
        summary = summarizer_model.summarize(nlp_text)
        summary_status = {
            "result": summary,
            "duration": (datetime.now() - step_start).total_seconds(),
            "status": "completed"
        }
    except Exception as e:
        summary_status = {
            "result": "",
            "duration": (datetime.now() - step_start).total_seconds(),
            "status": "failed",
            "error": str(e)
        }
    
    # Step 4: Insights (if enabled)
    insights = {}
    if include_insights:
        task_instance.update_state(
            state="PROCESSING",
            meta={"step": "insights", "progress": 95}
        )
        
        # Generate insights (simplified version)
        entities = ner_status["result"]
        classification = classifier_status["result"]
        summary = summary_status["result"]
        
        insights = _generate_insights(transcript, translation, entities, classification, summary)
    
    # Final result
    total_processing_time = (datetime.now() - start_time).total_seconds()
    
    result = {
        "audio_info": {
            "filename": filename,
            "file_size_mb": round(len(audio_bytes) / (1024 * 1024), 2),
            "language_specified": language,
            "processing_time": total_processing_time
        },
        "transcript": transcript,
        "translation": translation,
        "nlp_processing_info": {
            "text_used_for_nlp": nlp_source,
            "nlp_text_length": len(nlp_text)
        },
        "entities": ner_status["result"],
        "classification": classifier_status["result"],
        "summary": summary_status["result"],
        "insights": insights if include_insights else None,
        "processing_steps": {
            "transcription": processing_steps["transcription"],
            "translation": processing_steps.get("translation"),
            "ner": {
                "duration": ner_status["duration"],
                "status": ner_status["status"],
                "entities_found": len(ner_status["result"]) if ner_status["result"] else 0
            },
            "classification": {
                "duration": classifier_status["duration"],
                "status": classifier_status["status"],
                "confidence": classifier_status["result"].get("confidence", 0) if classifier_status["result"] else 0
            },
            "summarization": {
                "duration": summary_status["duration"],
                "status": summary_status["status"],
                "summary_length": len(summary_status["result"]) if summary_status["result"] else 0
            }
        },
        "pipeline_info": {
            "total_time": total_processing_time,
            "models_used": ["whisper"] + (["translator"] if include_translation else []) + ["ner", "classifier", "summarizer"],
            "text_flow": f"transcript → {nlp_source} → nlp_models",
            "timestamp": datetime.now().isoformat(),
            "processed_by": "celery_worker"
        }
    }
    
    return result

# Add the quick task with similar pattern
@celery_app.task(bind=True, name="process_audio_quick_task")
def process_audio_quick_task(
    self,
    audio_bytes: bytes,
    filename: str,
    language: Optional[str] = None
):
    """
    Celery task for quick audio analysis
    """
    try:
        self.update_state(
            state="PROCESSING",
            meta={
                "step": "quick_analysis",
                "filename": filename,
                "progress": 0
            }
        )
        
        # Get worker models
        models = get_worker_models()
        if not models:
            raise RuntimeError("Models not loaded in Celery worker")
        
        # Quick processing (no translation, no insights)
        result = _process_audio_sync_worker(
            self, models, audio_bytes, filename, language, 
            include_translation=False, include_insights=False
        )
        
        # Extract essentials for quick response
        classification = result.get("classification", {})
        insights = result.get("insights", {})
        risk_assessment = insights.get("risk_assessment", {}) if insights else {}
        
        quick_result = {
            "transcript": result["transcript"],
            "summary": result["summary"],
            "main_category": classification.get("main_category", "unknown"),
            "priority": classification.get("priority", "medium"),
            "risk_level": risk_assessment.get("risk_level", "unknown"),
            "processing_time": result["pipeline_info"]["total_time"]
        }
        
        return {
            "status": "completed",
            "filename": filename,
            "result": quick_result
        }
        
    except Exception as e:
        logger.error(f"Quick audio processing task failed: {e}")
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "filename": filename,
                "error_type": type(e).__name__
            }
        )
        raise type(e)(str(e))

def _generate_insights(transcript: str, translation: Optional[str], 
                      entities: Dict, classification: Dict, summary: str) -> Dict[str, Any]:
    """Generate basic insights from processed data"""
    
    persons = entities.get("PERSON", [])
    locations = entities.get("LOC", []) + entities.get("GPE", [])
    organizations = entities.get("ORG", [])
    dates = entities.get("DATE", [])
    
    primary_text = translation if translation else transcript
    
    # Basic risk assessment
    risk_keywords = ["suicide", "abuse", "violence", "threat", "danger", "crisis", "emergency"]
    risk_score = sum(1 for keyword in risk_keywords if keyword.lower() in primary_text.lower())
    
    return {
        "case_overview": {
            "primary_language": "multilingual" if translation else "original",
            "key_entities": {
                "people_mentioned": len(persons),
                "locations_mentioned": len(locations),
                "organizations_mentioned": len(organizations),
                "dates_mentioned": len(dates)
            },
            "case_complexity": "high" if len(persons) > 2 or len(locations) > 1 else "medium" if len(persons) > 0 else "low"
        },
        "risk_assessment": {
            "risk_indicators_found": risk_score,
            "risk_level": "high" if risk_score >= 2 else "medium" if risk_score >= 1 else "low",
            "priority": classification.get("priority", "medium"),
            "confidence": classification.get("confidence", 0)
        },
        "key_information": {
            "main_category": classification.get("main_category", "unknown"),
            "sub_category": classification.get("sub_category", "unknown"),
            "intervention_needed": classification.get("intervention", "assessment_required"),
            "summary": summary[:200] + "..." if len(summary) > 200 else summary
        },
        "entities_detail": {
            "persons": persons[:5],
            "locations": locations[:3],
            "organizations": organizations[:3],
            "key_dates": dates[:3]
        }
    }