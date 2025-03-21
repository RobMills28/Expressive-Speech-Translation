# services/espnet_backend.py
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .translation_strategy import TranslationBackend

logger = logging.getLogger(__name__)

class ESPnetBackend(TranslationBackend):
    """
    ESPnet-based translation backend
    """
    
    def __init__(self, device=None):
        """Initialize the ESPnet backend"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        self.languages = {
            'en': 'English',
            'fr': 'French', 
            'es': 'Spanish',
            'de': 'German',
            # Add more as you verify support
        }
        
        # Map between language codes
        self.language_map = {
            'eng': 'en',
            'fra': 'fr',
            'fre': 'fr',
            'spa': 'es',
            'deu': 'de',
            'ger': 'de',
            # Add more mappings
        }
        
        # Models will be initialized lazily
        self.asr_model = None
        self.mt_models = {}
        self.tts_models = {}
    
    def initialize(self):
        """Initialize models (lazy loading)"""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing ESPnet backend on {self.device}")
            
            # Will import and load models only when needed
            # This avoids loading unnecessary models at startup
            
            self.initialized = True
            logger.info("ESPnet backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ESPnet backend: {str(e)}")
            raise
    
    def _ensure_initialized(self):
        """Ensure backend is initialized before use"""
        if not self.initialized:
            self.initialize()
    
    def _load_asr_model(self):
        """Load ASR model if not already loaded"""
        if self.asr_model is not None:
            return
            
        try:
            logger.info("Loading ESPnet ASR model")
            from espnet2.bin.asr_inference import Speech2Text
            
            self.asr_model = Speech2Text.from_pretrained(
                model_tag="espnet/english_asr_conformer_transducer_large",
                device=self.device
            )
            logger.info("ESPnet ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ESPnet ASR model: {str(e)}")
            raise
    
    def _load_mt_model(self, src_lang: str, tgt_lang: str):
        """Load MT model for specific language pair if not already loaded"""
        model_key = f"{src_lang}-{tgt_lang}"
        if model_key in self.mt_models:
            return
            
        try:
            logger.info(f"Loading ESPnet MT model for {model_key}")
            from espnet2.bin.mt_inference import Text2Text
            
            self.mt_models[model_key] = Text2Text.from_pretrained(
                model_tag=f"espnet/{src_lang}_to_{tgt_lang}_transformer",
                device=self.device
            )
            logger.info(f"ESPnet MT model for {model_key} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ESPnet MT model for {model_key}: {str(e)}")
            raise
    
    def _load_tts_model(self, lang: str):
        """Load TTS model for specific language if not already loaded"""
        if lang in self.tts_models:
            return
            
        try:
            logger.info(f"Loading ESPnet TTS model for {lang}")
            from espnet2.bin.tts_inference import Text2Speech
            
            self.tts_models[lang] = Text2Speech.from_pretrained(
                model_tag=f"espnet/{lang}_tts_transformer",
                device=self.device
            )
            logger.info(f"ESPnet TTS model for {lang} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ESPnet TTS model for {lang}: {str(e)}")
            raise
    
    def translate_speech(
        self, 
        audio_tensor: torch.Tensor, 
        source_lang: str = "en",
        target_lang: str = "fr"
    ) -> Dict[str, Any]:
        """
        Translate speech using ESPnet
        
        Args:
            audio_tensor: Processed audio tensor
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Dictionary with translated audio, source text, and target text
        """
        self._ensure_initialized()
        
        # Map language codes if needed
        src_lang = self.language_map.get(source_lang, source_lang)
        tgt_lang = self.language_map.get(target_lang, target_lang)
        
        try:
            # Step 1: Speech recognition (ASR)
            self._load_asr_model()
            audio_numpy = audio_tensor.squeeze().numpy()
            
            # Run ASR
            asr_result = self.asr_model(audio_numpy)[0]
            source_text = asr_result[0]
            
            # Step 2: Text translation (MT)
            self._load_mt_model(src_lang, tgt_lang)
            
            # Run MT
            model_key = f"{src_lang}-{tgt_lang}"
            mt_result = self.mt_models[model_key](source_text)
            translated_text = mt_result[0]
            
            # Step 3: Text to speech (TTS)
            self._load_tts_model(tgt_lang)
            
            # Run TTS
            tts_output = self.tts_models[tgt_lang](translated_text)
            translated_audio = tts_output["wav"]
            
            # Convert numpy array to torch tensor
            output_tensor = torch.from_numpy(translated_audio).unsqueeze(0)
            
            return {
                "audio": output_tensor,
                "transcripts": {
                    "source": source_text,
                    "target": translated_text
                }
            }
            
        except Exception as e:
            logger.error(f"ESPnet translation failed: {str(e)}")
            raise
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported"""
        # Normalize language code
        lang = self.language_map.get(language_code, language_code)
        return lang in self.languages
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        return self.languages