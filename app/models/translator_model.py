import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class TranslationModel:
    """Translation model using HuggingFace Transformers"""

    def __init__(self, model_path: str = "/opt/chl_ai/models/translation/finetuned-sw-en/checkpoint-513920"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.load_time = None
        self.error = None

    def load(self) -> bool:
        try:
            logger.info(f"Loading translation model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.loaded = True
            self.load_time = datetime.now()
            logger.info("✅ Translation model loaded successfully")
            return True
        except Exception as e:
            self.error = str(e)
            logger.error(f"❌ Failed to load translation model: {e}")
            return False

    def translate(self, text: str) -> str:
        if not self.loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Translation model not loaded")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_model_info(self) -> Dict:
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "loaded": self.loaded,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "error": self.error
        }

# Global instance
translator_model = TranslationModel()
