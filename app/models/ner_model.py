import spacy
import logging
from typing import Dict, List, Union, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class NERModel:
    """Named Entity Recognition using spaCy"""
    
    def __init__(self, model_name: str = "en_core_web_md"):
        self.model_name = model_name
        self.nlp = None
        self.loaded = False
        self.load_time = None
        self.error = None
        
    def load(self) -> bool:
        """Load the spaCy NER model"""
        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            start_time = datetime.now()
            
            # Load spaCy model
            self.nlp = spacy.load(self.model_name)
            
            # Test the model with a simple example
            test_doc = self.nlp("Test loading with Barack Obama in Washington.")
            if len(test_doc.ents) == 0:
                logger.warning("Model loaded but no entities detected in test")
            
            self.loaded = True
            self.load_time = datetime.now()
            load_duration = (self.load_time - start_time).total_seconds()
            
            logger.info(f"spaCy NER model loaded successfully in {load_duration:.2f}s")
            return True
            
        except Exception as e:
            self.error = str(e)
            self.load_time = datetime.now()
            logger.error(f"Failed to load spaCy model: {e}")
            return False
    
    def extract_entities(self, text: str, flat: bool = True) -> Union[Dict[str, List[str]], List[Dict[str, str]]]:
        """
        Extract named entities from text
        
        Args:
            text (str): Input text
            flat (bool): If True, return flat list. If False, return grouped by entity type
        
        Returns:
            Union[Dict, List]: Entities in requested format
        """
        if not self.loaded or self.nlp is None:
            raise RuntimeError("NER model not loaded. Call load() first.")
        
        if not text or not text.strip():
            return [] if flat else {}
        
        try:
            # Process text with spaCy
            doc = self.nlp(text.strip())
            
            if flat:
                # Return flat list of entities
                entities = []
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": getattr(ent, 'confidence', 1.0)  # spaCy doesn't always have confidence
                    })
                return entities
            else:
                # Return entities grouped by type
                entity_dict = {}
                for ent in doc.ents:
                    if ent.label_ not in entity_dict:
                        entity_dict[ent.label_] = []
                    entity_dict[ent.label_].append(ent.text)
                return entity_dict
                
        except Exception as e:
            logger.error(f"NER processing failed: {e}")
            raise RuntimeError(f"NER processing failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            "model_name": self.model_name,
            "loaded": self.loaded,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "error": self.error
        }
        
        if self.loaded and self.nlp:
            info.update({
                "spacy_version": spacy.__version__,
                "model_version": self.nlp.meta.get("version", "unknown"),
                "language": self.nlp.meta.get("lang", "unknown"),
                "pipeline": list(self.nlp.pipe_names),
                "labels": list(self.nlp.get_pipe("ner").labels) if "ner" in self.nlp.pipe_names else []
            })
        
        return info
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self.loaded and self.nlp is not None and self.error is None

# Global NER model instance
ner_model = NERModel()