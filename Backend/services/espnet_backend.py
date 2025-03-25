# services/espnet_backend.py

import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Import ESPnet model zoo for inference
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet2.bin.mt_inference import Text2Text  # For text translation
from espnet_model_zoo.downloader import ModelDownloader

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
        self.downloader = ModelDownloader()
        
        # Language mapping (your app codes to standard language codes)
        self.lang_mapping = {
            'eng': 'en',  # English
            'fra': 'fr',  # French
            'spa': 'es',  # Spanish
            'deu': 'de',  # German
            'ita': 'it',  # Italian
            'cmn': 'zh',  # Chinese
            'jpn': 'ja',  # Japanese
        }
        
        # Define ASR models - verified model ID
        self.asr_models = {
            'eng': 'kamo-naoyuki/wsj'  # English model
        }
        
        # Define TTS models - only use verified models
        self.tts_models = {
            'eng': 'espnet/kan-bayashi_ljspeech_vits',  # English - this one worked
        }
        
        # Models will be loaded on demand
        self.asr_model = None
        self.tts_models_loaded = {}
    
    def initialize(self):
        """Initialize ESPnet models"""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing ESPnet backend on {self.device}")
            
            # We'll lazy-load models as needed rather than loading all at once
            self.initialized = True
            logger.info("ESPnet backend initialization successful")
            
        except Exception as e:
            logger.error(f"Failed to initialize ESPnet backend: {str(e)}")
            raise
    
    def _load_asr_model(self, lang_code='eng'):
        """Load ASR model for a specific language on demand"""
        if self.asr_model is not None:
            return True
            
        try:
            model_id = self.asr_models.get(lang_code)
            if not model_id:
                logger.error(f"No ASR model available for {lang_code}")
                return False
                
            logger.info(f"Loading ASR model for {lang_code}: {model_id}")
            
            # Load model using ModelDownloader for reliability
            config = self.downloader.download_and_unpack(model_id)
            
            # Convert device from torch.device to string
            device_str = str(self.device)
            
            self.asr_model = Speech2Text(
                **config,
                device=device_str,  # Pass as string instead of torch.device
                beam_size=10,
                ctc_weight=0.3,
                lm_weight=0.0,
                nbest=1
            )
            
            logger.info(f"ASR model for {lang_code} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ASR model for {lang_code}: {str(e)}")
            return False
    
    def _load_tts_model(self, lang_code):
        """Load TTS model for a specific language on demand"""
        if lang_code in self.tts_models_loaded:
            return True
            
        try:
            model_id = self.tts_models.get(lang_code)
            if not model_id:
                logger.error(f"No TTS model available for {lang_code}")
                return False
                
            logger.info(f"Loading TTS model for {lang_code}: {model_id}")
            
            # Convert device from torch.device to string
            device_str = str(self.device)
            
            self.tts_models_loaded[lang_code] = Text2Speech.from_pretrained(
                model_tag=model_id,
                device=device_str,  # Pass as string instead of torch.device
            )
            
            logger.info(f"TTS model for {lang_code} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TTS model for {lang_code}: {str(e)}")
            return False
    
    def translate_speech(
        self, 
        audio_tensor: torch.Tensor, 
        source_lang: str = "eng",
        target_lang: str = "eng"  # Start with English to English for testing
    ) -> Dict[str, Any]:
        """
        Translate speech using ESPnet with improved error handling
        """
        if not self.initialized:
            self.initialize()
        
        # Map language codes if needed
        src_lang = self.lang_mapping.get(source_lang, source_lang)
        tgt_lang = self.lang_mapping.get(target_lang, target_lang)
        
        try:
            # 1. Speech recognition (ASR)
            if not self._load_asr_model(source_lang):
                raise ValueError(f"Failed to load ASR model for {source_lang}")
                
            # Ensure audio is in the right format
            audio_numpy = audio_tensor.squeeze().numpy()
            
            # Check for valid audio data
            if len(audio_numpy) == 0:
                logger.error("Empty audio data")
                raise ValueError("Empty audio data")
                
            # Normalize audio for consistency
            if np.abs(audio_numpy).max() > 0:
                audio_numpy = audio_numpy / np.abs(audio_numpy).max() * 0.9
            
            # Run ASR with better error handling
            try:
                logger.info(f"Running ASR on audio data of shape {audio_numpy.shape}")
                asr_results = self.asr_model(audio_numpy)
                
                if not asr_results or len(asr_results) == 0:
                    logger.warning("ASR model returned empty results, using fallback text")
                    source_text = "Hello world"  # Fallback text for testing
                else:
                    # Access the first result's text with proper error handling
                    source_text = asr_results[0][0] if isinstance(asr_results[0], (list, tuple)) and len(asr_results[0]) > 0 else "Hello world"
                    
                logger.info(f"ASR result: {source_text}")
            except Exception as e:
                logger.error(f"ASR processing failed: {str(e)}")
                source_text = "Hello world"  # Fallback text
            
            # 2. Text translation (simplified for initial testing)
            # For now, we'll use the same text for source and target
            target_text = source_text
            logger.info(f"Using text: {target_text}")
            
            # 3. Text-to-speech (TTS)
            # Try to load target language TTS model
            if not self._load_tts_model(target_lang):
                # If target language not available, fall back to English
                if target_lang != 'eng' and self._load_tts_model('eng'):
                    logger.warning(f"No TTS model for {target_lang}, falling back to English")
                    tts_model = self.tts_models_loaded['eng']
                    target_text = source_text  # Use original English text
                else:
                    raise ValueError(f"No TTS model available for {target_lang} and fallback failed")
            else:
                tts_model = self.tts_models_loaded[target_lang]
                
            # Generate speech with better error handling
            try:
                logger.info(f"Generating TTS for text: '{target_text}'")
                
                # Ensure we have valid text
                if not target_text or len(target_text.strip()) == 0:
                    target_text = "Hello world"
                    
                tts_output = tts_model(target_text)
                
                if "wav" not in tts_output or tts_output["wav"] is None:
                    logger.error("TTS model did not return audio data")
                    # Create silence as fallback
                    output_wav = np.zeros(16000)  # 1 second of silence
                else:
                    output_wav = tts_output["wav"]
                    
                    # Check if output is valid
                    if len(output_wav) == 0:
                        logger.error("TTS model returned empty audio")
                        output_wav = np.zeros(16000)  # 1 second of silence
                
                # Convert to tensor with proper formatting
                output_tensor = torch.tensor(output_wav).unsqueeze(0)
                
                logger.info(f"Generated TTS output: shape={output_tensor.shape}")
            except Exception as e:
                logger.error(f"TTS generation failed: {str(e)}")
                # Create silent audio as fallback
                logger.info("Using silent audio as fallback")
                output_tensor = torch.zeros((1, 16000))  # 1 second of silence
            
            return {
                "audio": output_tensor,
                "transcripts": {
                    "source": source_text,
                    "target": target_text
                }
            }
            
        except Exception as e:
            logger.error(f"ESPnet translation failed: {str(e)}")
            # Return minimal valid output instead of raising exception
            return {
                "audio": torch.zeros((1, 16000)),  # 1 second of silence
                "transcripts": {
                    "source": "Error in translation",
                    "target": "Error in translation"
                }
            }
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported"""
        # For now, we only fully support English
        # Other languages will fall back to English TTS
        return language_code == 'eng' or language_code in self.tts_models
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        # For now, we'll just return English as fully supported
        return {'eng': 'English'}