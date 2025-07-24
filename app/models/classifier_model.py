import logging
from typing import Dict
from datetime import datetime
import torch
import re
import json
from transformers import AutoTokenizer
from transformers import DistilBertPreTrainedModel, DistilBertModel
import torch.nn as nn

logger = logging.getLogger(__name__)

# Load category lists
with open("/opt/chl_ai/models/ai_models/MultiClassifier/main_categories.json") as f:
    main_categories = json.load(f)
with open("/opt/chl_ai/models/ai_models/MultiClassifier/sub_categories.json") as f:
    sub_categories = json.load(f)
with open("/opt/chl_ai/models/ai_models/MultiClassifier/interventions.json") as f:
    interventions = json.load(f)
with open("/opt/chl_ai/models/ai_models/MultiClassifier/priorities.json") as f:
    priorities = json.load(f)

class MultiTaskDistilBert(DistilBertPreTrainedModel):
    """Custom DistilBERT model for multi-task classification"""
    
    def __init__(self, config, num_main, num_sub, num_interv, num_priority):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier_main = nn.Linear(config.dim, num_main)
        self.classifier_sub = nn.Linear(config.dim, num_sub)
        self.classifier_interv = nn.Linear(config.dim, num_interv)
        self.classifier_priority = nn.Linear(config.dim, num_priority)
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_state = distilbert_output.last_hidden_state 
        pooled_output = hidden_state[:, 0]                 
        pooled_output = self.pre_classifier(pooled_output) 
        pooled_output = nn.ReLU()(pooled_output)           
        pooled_output = self.dropout(pooled_output)        
        
        logits_main = self.classifier_main(pooled_output)
        logits_sub = self.classifier_sub(pooled_output)
        logits_interv = self.classifier_interv(pooled_output)
        logits_priority = self.classifier_priority(pooled_output)
        
        return (logits_main, logits_sub, logits_interv, logits_priority)

class ClassifierModel:
    """Multi-task classifier for case narratives"""
    
    # def __init__(self, model_path: str = './ai_models/MultiClassifier/multitask_distilbert/'):
    def __init__(self, model_path: str = '/opt/chl_ai/models/ai_models/MultiClassifier/multitask_distilbert'):

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.load_time = None
        self.error = None
        self.category_info = {
            "main_categories": main_categories,
            "sub_categories": sub_categories,
            "interventions": interventions,
            "priorities": priorities
        }

    def load(self) -> bool:
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading classifier model: {self.model_path}")
            start_time = datetime.now()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = MultiTaskDistilBert.from_pretrained(
                self.model_path,
                num_main=len(main_categories),
                num_sub=len(sub_categories),
                num_interv=len(interventions),
                num_priority=len(priorities)
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # ✅ Only mark as loaded here
            self.loaded = True
            
            # Optional test after loaded flag is True
            try:
                test_result = self.classify("Test classification model loading")
                logger.debug(f"Test classification result: {test_result}")
            except Exception as test_error:
                logger.warning(f"Model test failed: {test_error}")
            
            self.load_time = datetime.now()
            load_duration = (self.load_time - start_time).total_seconds()
            logger.info(f"Classifier model loaded successfully in {load_duration:.2f}s")
            return True
            
        except Exception as e:
            self.error = str(e)
            self.load_time = datetime.now()
            logger.error(f"Failed to load classifier model: {e}")
            return False

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        text = text.lower().strip()
        return re.sub(r'[^a-z0-9\s]', '', text)
    
    def classify(self, narrative: str) -> Dict[str, str]:
        """
        Classify case narrative into categories
        
        Args:
            narrative (str): Input case description
            
        Returns:
            Dict: Classification results with keys:
                main_category, sub_category, intervention, priority
        """
        if not self.loaded or not self.tokenizer or not self.model:
            raise RuntimeError("Classifier model not loaded. Call load() first.")
        
        if not narrative or not narrative.strip():
            return {}
        
        try:
            # Preprocess and tokenize
            clean_text = self.preprocess_text(narrative)
            inputs = self.tokenizer(
                clean_text,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(**inputs)
                logits_main, logits_sub, logits_interv, logits_priority = logits
            
            # Process predictions
            preds_main = torch.argmax(logits_main, dim=1).item()
            preds_sub = torch.argmax(logits_sub, dim=1).item()
            preds_interv = torch.argmax(logits_interv, dim=1).item()
            preds_priority = torch.argmax(logits_priority, dim=1).item()
            
            return {
                "main_category": main_categories[preds_main],
                "sub_category": sub_categories[preds_sub],
                "intervention": interventions[preds_interv],
                "priority": str(priorities[preds_priority])
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise RuntimeError(f"Classification failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            "model_path": self.model_path,
            "loaded": self.loaded,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "device": str(self.device),
            "error": self.error,
            "num_categories": {
                "main": len(main_categories),
                "sub": len(sub_categories),
                "intervention": len(interventions),
                "priority": len(priorities)
            }
        }
        
        if self.loaded and self.model:
            info.update({
                "model_type": type(self.model).__name__,
                "tokenizer": type(self.tokenizer).__name__
            })
        
        return info
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self.loaded and self.tokenizer is not None and self.model is not None

# Global classifier model instance
classifier_model = ClassifierModel()