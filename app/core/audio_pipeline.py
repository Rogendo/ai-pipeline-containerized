import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .resource_manager import resource_manager
from .request_queue import request_queue
from ..models.model_loader import model_loader

logger = logging.getLogger(__name__)

class AudioPipelineService:
    """Orchestrates complete audio-to-insights pipeline"""
    
    def __init__(self):
        self.is_ready = False
        
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
    async def process_audio_complete(
        self, 
        audio_bytes: bytes, 
        filename: str,
        language: Optional[str] = None,
        include_translation: bool = True,
        include_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Complete audio processing pipeline with all models
        
        Flow: Audio → Whisper → Translation → NLP Models (NER + Classifier + Summarizer) → Insights
        """
        
        # Check pipeline readiness
        readiness = self.check_pipeline_readiness()
        if not readiness["pipeline_ready"]:
            raise RuntimeError(f"Pipeline not ready. Missing models: {readiness['missing_models']}")
        
        start_time = datetime.now()
        processing_steps = {}
        
        try:
            # Step 1: Audio Transcription (GPU intensive - use resource manager)
            logger.info("🎙️ Starting audio transcription...")
            step_start = datetime.now()
            
            async with resource_manager.gpu_semaphore:  # Use existing resource management
                whisper_model = model_loader.models.get("whisper")
                transcript = whisper_model.transcribe_audio_bytes(audio_bytes, language=language)
            
            processing_steps["transcription"] = {
                "duration": (datetime.now() - step_start).total_seconds(),
                "status": "completed",
                "output_length": len(transcript)
            }
            logger.info(f"✅ Transcription completed: {len(transcript)} characters")
            
            # Step 2: Translation (Sequential - must complete before NLP models)
            translation = None
            if include_translation:
                logger.info("🌐 Starting translation...")
                step_start = datetime.now()
                
                try:
                    translator_model = model_loader.models.get("translator")
                    translation = translator_model.translate(transcript)
                    
                    processing_steps["translation"] = {
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed",
                        "output_length": len(translation)
                    }
                    logger.info(f"✅ Translation completed: {len(translation)} characters")
                    
                except Exception as e:
                    logger.error(f"❌ Translation failed: {e}")
                    processing_steps["translation"] = {
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "failed",
                        "error": str(e)
                    }
                    # Continue with original transcript if translation fails
                    translation = None
            
            # Step 3: Determine text for NLP models
            # Use translated text if available, otherwise use original transcript
            nlp_text = translation if translation else transcript
            nlp_source = "translated_text" if translation else "original_transcript"
            
            logger.info(f"🧠 Starting NLP analysis on {nlp_source}...")
            
            # Step 4: Parallel NLP Processing (CPU based - can run concurrently)
            # All NLP models work on the same text (translated or original)
            async def run_ner():
                step_start = datetime.now()
                try:
                    ner_model = model_loader.models.get("ner")
                    entities = ner_model.extract_entities(nlp_text, flat=False)  # Using nlp_text
                    return {
                        "result": entities,
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed"
                    }
                except Exception as e:
                    logger.error(f"❌ NER failed: {e}")
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
                    classification = classifier_model.classify(nlp_text)  # Using nlp_text
                    return {
                        "result": classification,
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed"
                    }
                except Exception as e:
                    logger.error(f"❌ Classification failed: {e}")
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
                    summary = summarizer_model.summarize(nlp_text)  # Using nlp_text
                    return {
                        "result": summary,
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "completed"
                    }
                except Exception as e:
                    logger.error(f"❌ Summarization failed: {e}")
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
            
            logger.info(f"✅ NLP processing completed using {nlp_source}")
            
            # Step 5: Generate insights (if enabled)
            insights = {}
            if include_insights:
                logger.info("🔍 Generating case insights...")
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
                    logger.warning(f"⚠️ Insights generation failed: {e}")
                    processing_steps["insights"] = {
                        "duration": (datetime.now() - step_start).total_seconds(),
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Calculate total processing time
            total_processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build complete response
            result = {
                "audio_info": {
                    "filename": filename,
                    "file_size_mb": round(len(audio_bytes) / (1024 * 1024), 2),
                    "language_specified": language,
                    "processing_time": total_processing_time
                },
                "transcript": transcript,  # Original language
                "translation": translation,  # English (if enabled)
                "nlp_processing_info": {
                    "text_used_for_nlp": nlp_source,
                    "nlp_text_length": len(nlp_text)
                },
                "entities": entities,  # From translated text (or original if no translation)
                "classification": classification,  # From translated text (or original if no translation)
                "summary": summary,  # From translated text (or original if no translation)
                "insights": insights if include_insights else None,
                "processing_steps": processing_steps,
                "pipeline_info": {
                    "total_time": total_processing_time,
                    "models_used": ["whisper"] + (["translator"] if include_translation else []) + ["ner", "classifier", "summarizer"],
                    "text_flow": f"transcript → {nlp_source} → nlp_models",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"🎉 Complete audio pipeline finished in {total_processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Audio pipeline failed: {e}")
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