# app/core/audio_pipeline.py (Enhanced)
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from .resource_manager import resource_manager
from .request_queue import request_queue, RequestStatus
from ..models.model_loader import model_loader

logger = logging.getLogger(__name__)


class AudioPipelineService:
    """Orchestrates complete audio-to-insights pipeline with queue integration"""
    
    def __init__(self):
        self.is_ready = False
        self.processing_tasks: Dict[str, asyncio.Task] = {}
    
    def check_pipeline_readiness(self) -> Dict[str, Any]:
        """Check if all required models are ready"""
        required_models = ["whisper", "ner", "classifier_model", "translator", "summarizer"]
        model_status = {}
        all_ready = True
        
        for model_name in required_models:
            is_ready = model_loader.is_model_ready(model_name)
            model_status[model_name] = is_ready
            if not is_ready:
                all_ready = False
        
        self.is_ready = all_ready
        
        return {
            "pipeline_ready": all_ready,
            "models": model_status,
            "missing_models": [name for name, ready in model_status.items() if not ready]
        }
    async def submit_audio_request(
        self, 
        audio_bytes: bytes, 
        filename: str,
        language: Optional[str] = None,
        include_translation: bool = True,
        include_insights: bool = True,
        background: bool = True
    ) -> Dict[str, Any]:
        """
        Submit audio processing request - returns immediately with request_id if background=True
        """
        
        # Check pipeline readiness
        readiness = self.check_pipeline_readiness()
        if not readiness["pipeline_ready"]:
            raise RuntimeError(f"Pipeline not ready. Missing models: {readiness['missing_models']}")
        
        if background:
            # Background processing - return request_id immediately
            request_id = await request_queue.add_request("audio_processing", priority=5)
            
            # Store request data (in production, use Redis or database)
            request_data = {
                "audio_bytes": audio_bytes,
                "filename": filename,
                "language": language,
                "include_translation": include_translation,
                "include_insights": include_insights
            }
            
            # Start background processing
            asyncio.create_task(
                self._process_audio_background(request_id, request_data)
            )
            # self.processing_tasks[request_id] = task
            
            return {
                "request_id": request_id,
                "status": "queued",
                "message": "Audio processing started. Check status at /queue/status/{request_id}",
                "estimated_time": "15-45 seconds",
                "status_endpoint": f"/queue/status/{request_id}"
            }
        else:
            # Synchronous processing with queue integration
            request_id = await request_queue.add_request("audio_processing_sync", priority=1)
            
            try:
                result = await self._process_audio_with_queue(
                    request_id, audio_bytes, filename, language, 
                    include_translation, include_insights
                )
                request_queue.complete_request(request_id, result=result)
                return result
                
            except Exception as e:
                request_queue.complete_request(request_id, error=str(e))
                raise
    
    # In _process_audio_background method
    async def _process_audio_background(self, request_id: str, request_data: Dict):
        logger.info(f"🔄 [{request_id}] Background processing started")
        
        try:
            # Update status to processing
            request_queue.update_request_status(request_id, "processing")
            
            result = await self._process_audio_with_queue(...)
            
            # Mark as completed
            logger.info(f"✅ [{request_id}] Completing request with result")
            request_queue.complete_request(request_id, result=result)
            logger.info(f"✅ [{request_id}] Request marked as completed")
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Background processing failed: {e}")
            request_queue.complete_request(request_id, error=str(e))
        
        # Verify final status
        final_status = request_queue.get_request_status(request_id)
        logger.info(f"🔍 [{request_id}] Final status: {final_status}")
        
        # finally:
        #     # Cleanup
        #     if request_id in self.processing_tasks:
        #         del self.processing_tasks[request_id]
    
    async def _process_audio_with_queue(
        self, 
        request_id: str,
        audio_bytes: bytes, 
        filename: str,
        language: Optional[str] = None,
        include_translation: bool = True,
        include_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio with proper queue and resource management integration
        """
        
        start_time = datetime.now()
        processing_steps = {}
        
        try:
            # Update queue status - starting processing
            if request_id in request_queue.requests:
                request_queue.requests[request_id].status = RequestStatus.PROCESSING
            
            # Step 1: Audio Transcription (GPU intensive - use resource manager)
            logger.info(f"🎙️ [{request_id}] Starting audio transcription...")
            step_start = datetime.now()
            
            # Proper resource management integration
            if not await resource_manager.acquire_gpu(request_id):
                raise RuntimeError("Failed to acquire GPU resources")
            
            try:
                whisper_model = model_loader.models.get("whisper")
                transcript = whisper_model.transcribe_audio_bytes(audio_bytes, language=language)
            finally:
                resource_manager.release_gpu(request_id)
            
            processing_steps["transcription"] = {
                "duration": (datetime.now() - step_start).total_seconds(),
                "status": "completed",
                "output_length": len(transcript)
            }
            logger.info(f"✅ [{request_id}] Transcription completed: {len(transcript)} characters")
            
            # Step 2: Translation (Sequential)
            translation = None
            if include_translation:
                logger.info(f"🌐 [{request_id}] Starting translation...")
                step_start = datetime.now()
                
                try:
                    translator_model = model_loader.models.get("translator")
                    translation = translator_model.translate(transcript)
                    
                    processing_steps["translation"] = {
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed",
                        "output_length": len(translation)
                    }
                    logger.info(f"✅ [{request_id}] Translation completed: {len(translation)} characters")
                    
                except Exception as e:
                    logger.error(f"❌ [{request_id}] Translation failed: {e}")
                    processing_steps["translation"] = {
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "failed",
                        "error": str(e)
                    }
                    translation = None
            
            # Step 3: Determine text for NLP models
            nlp_text = translation if translation else transcript
            nlp_source = "translated_text" if translation else "original_transcript"
            
            logger.info(f"🧠 [{request_id}] Starting NLP analysis on {nlp_source}...")
            
            # Step 4: Parallel NLP Processing (CPU based)
            async def run_ner():
                step_start = datetime.now()
                try:
                    ner_model = model_loader.models.get("ner")
                    entities = ner_model.extract_entities(nlp_text, flat=False)
                    return {
                        "result": entities,
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed"
                    }
                except Exception as e:
                    logger.error(f"❌ [{request_id}] NER failed: {e}")
                    return {
                        "result": {},
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "failed",
                        "error": str(e)
                    }
            
            async def run_classifier():
                step_start = datetime.now()
                try:
                    classifier_model = model_loader.models.get("classifier_model")
                    classification = classifier_model.classify(nlp_text)
                    return {
                        "result": classification,
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed"
                    }
                except Exception as e:
                    logger.error(f"❌ [{request_id}] Classification failed: {e}")
                    return {
                        "result": {},
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "failed",
                        "error": str(e)
                    }
            
            async def run_summarization():
                step_start = datetime.now()
                try:
                    summarizer_model = model_loader.models.get("summarizer")
                    summary = summarizer_model.summarize(nlp_text)
                    return {
                        "result": summary,
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed"
                    }
                except Exception as e:
                    logger.error(f"❌ [{request_id}] Summarization failed: {e}")
                    return {
                        "result": "",
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Run all NLP tasks in parallel
            ner_task, classifier_task, summary_task = await asyncio.gather(
                run_ner(),
                run_classifier(), 
                run_summarization(),
                return_exceptions=True
            )
            
            # Extract results
            entities = ner_task["result"] if isinstance(ner_task, dict) else {}
            classification = classifier_task["result"] if isinstance(classifier_task, dict) else {}
            summary = summary_task["result"] if isinstance(summary_task, dict) else ""
            
            # Log processing steps
            processing_steps.update({
                "ner": {
                    "duration": ner_task.get("duration", 0),
                    "status": ner_task.get("status", "failed"),
                    "entities_found": len(entities) if entities else 0,
                    "text_source": nlp_source
                },
                "classification": {
                    "duration": classifier_task.get("duration", 0),
                    "status": classifier_task.get("status", "failed"),
                    "confidence": classification.get("confidence", 0) if classification else 0,
                    "text_source": nlp_source
                },
                "summarization": {
                    "duration": summary_task.get("duration", 0),
                    "status": summary_task.get("status", "failed"),
                    "summary_length": len(summary) if summary else 0,
                    "text_source": nlp_source
                }
            })
            
            logger.info(f"✅ [{request_id}] NLP processing completed using {nlp_source}")
            
            # Step 5: Generate insights (if enabled)
            insights = {}
            if include_insights:
                logger.info(f"🔍 [{request_id}] Generating case insights...")
                step_start = datetime.now()
                
                try:
                    insights = self._generate_insights(
                        transcript, translation, entities, classification, summary
                    )
                    processing_steps["insights"] = {
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed"
                    }
                except Exception as e:
                    logger.warning(f"⚠️ [{request_id}] Insights generation failed: {e}")
                    processing_steps["insights"] = {
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Calculate total processing time
            total_processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build complete response
            result = {
                "request_id": request_id,
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
                "entities": entities,
                "classification": classification,
                "summary": summary,
                "insights": insights if include_insights else None,
                "processing_steps": processing_steps,
                "pipeline_info": {
                    "total_time": total_processing_time,
                    "models_used": ["whisper"] + (["translator"] if include_translation else []) + ["ner", "classifier", "summarizer"],
                    "text_flow": f"transcript → {nlp_source} → nlp_models",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"🎉 [{request_id}] Complete audio pipeline finished in {total_processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Audio pipeline failed: {e}")
            raise RuntimeError(f"Audio pipeline processing failed: {str(e)}")
        
    
    def _generate_insights(self, transcript: str, translation: Optional[str], 
                          entities: Dict, classification: Dict, summary: str) -> Dict[str, Any]:
        """Generate case insights from all processed data"""
        
        # Extract key information
        persons = entities.get("PERSON", [])
        locations = entities.get("LOC", []) + entities.get("GPE", [])
        organizations = entities.get("ORG", [])
        dates = entities.get("DATE", [])
        
        # Determine primary language
        primary_text = translation if translation else transcript
        
        # Calculate risk indicators
        risk_keywords = ["suicide", "abuse", "violence", "threat", "danger", "crisis", "emergency"]
        risk_score = sum(1 for keyword in risk_keywords if keyword.lower() in primary_text.lower())
        
        # Generate insights
        insights = {
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
                "persons": persons[:5],  # Limit to first 5
                "locations": locations[:3],
                "organizations": organizations[:3],
                "key_dates": dates[:3]
            }
        }
        
        return insights

# Global instance
audio_pipeline = AudioPipelineService()