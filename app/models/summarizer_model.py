# app/models/summarizer_model.py

import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class SummarizationModel:
    """Summarization model using HuggingFace Transformers"""

    def __init__(self, model_path: str = "/opt/chl_ai/models/summarization/best_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.load_time = None
        self.error = None

    def load(self) -> bool:
        try:
            logger.info(f"📦 Loading summarization model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True)
            self.model.to(self.device)
            self.pipeline = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
            self.loaded = True
            self.load_time = datetime.now()
            logger.info("✅ Summarization model loaded successfully")
            return True
        except Exception as e:
            self.error = str(e)
            logger.error(f"❌ Failed to load summarization model: {e}")
            return False

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        if not self.loaded or self.pipeline is None:
            raise RuntimeError("Summarization model not loaded")

        try:
            logger.debug(f"📝 Summarizing text: {text[:100]}...")
            summary = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            raise

    def get_model_info(self) -> Dict:
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "loaded": self.loaded,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "error": self.error,
            "task": "text-summarization",
            "framework": "transformers",
        }


# Global instance
summarization_model = SummarizationModel()
