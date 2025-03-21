# services/seamless_backend.py
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from transformers import (
    SeamlessM4TProcessor, 
    SeamlessM4Tv2Model,
    SeamlessM4Tv2ForSpeechToText
)

from .translation_strategy import TranslationBackend

logger = logging.getLogger(__name__)

class SeamlessBackend(TranslationBackend):
    """
    SeamlessM4T-based translation backend
    """
    
    def __init__(self, device=None, auth_token=None):
        """Initialize the Seamless backend"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.auth_token = auth_token or os.getenv('HUGGINGFACE_TOKEN')
        self.model_name = "facebook/seamless-m4t-v2-large"
        self.initialized = False
        
        # Seamless supported languages
        self.languages = {
            'eng': 'English',
            'fra': 'French',
            'spa': 'Spanish',
            'deu': 'German',
            'ita': 'Italian',
            'por': 'Portuguese',
            'rus': 'Russian',
            'cmn': 'Chinese (Simplified)',
            'jpn': 'Japanese',
            # Add more from your LANGUAGE_CODES
        }
        
        # Models will be loaded during initialization
        self.processor = None
        self.model = None
        self.text_model = None
        self.tokenizer = None
    
    def initialize(self):
        """Initialize Seamless models"""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing Seamless backend on {self.device}")
            
            # Load processor
            self.processor = SeamlessM4TProcessor.from_pretrained(
                self.model_name, 
                token=self.auth_token,
                trust_remote_code=True
            )
            
            # Load main model
            self.model = SeamlessM4Tv2Model.from_pretrained(
                self.model_name, 
                token=self.auth_token,
                torch_dtype=torch.float32,
            )
            
            # Load text model
            self.text_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                self.model_name,
                token=self.auth_token,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            
            # Move models to device
            if self.device.type == 'cuda':
                self.model = self.model.to(self.device)
                self.text_model = self.text_model.to(self.device)
                
                # Set models to eval mode
                self.model.eval()
                self.text_model.eval()
                
                logger.info(f"Models loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.model.eval()
                self.text_model.eval()
                logger.info("Models loaded on CPU")
            
            self.initialized = True
            logger.info("Seamless backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Seamless backend: {str(e)}")
            raise
    
    def translate_speech(
        self, 
        audio_tensor: torch.Tensor, 
        source_lang: str = "eng",
        target_lang: str = "fra"
    ) -> Dict[str, Any]:
        """
        Translate speech using Seamless M4T
        
        Args:
            audio_tensor: Processed audio tensor
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Dictionary with translated audio, source text, and target text
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Convert tensor to numpy array
            audio_numpy = audio_tensor.squeeze().numpy()
            
            # Prepare inputs
            inputs = self.processor(
                audios=audio_numpy,
                sampling_rate=16000,
                return_tensors="pt",
                src_lang=source_lang,
                tgt_lang=target_lang,
                padding=True,
                max_length=512000
            )
            
            if self.device.type == 'cuda':
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            
            # Generate source text
            with torch.no_grad():
                source_outputs = self.text_model.generate(
                    input_features=inputs["input_features"],
                    tgt_lang="eng",
                    num_beams=6,
                    do_sample=False,
                    max_new_tokens=8000,
                    temperature=0.2,
                    length_penalty=2.0,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3
                )
                source_text = self.processor.batch_decode(source_outputs, skip_special_tokens=True)[0]
            
            # Generate target text
            with torch.no_grad():
                target_outputs = self.text_model.generate(
                    input_features=inputs["input_features"],
                    tgt_lang=target_lang,
                    num_beams=6,
                    max_new_tokens=8000,
                    length_penalty=2.0,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3
                )
                target_text = self.processor.batch_decode(target_outputs, skip_special_tokens=True)[0]
            
            # Generate translated audio
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tgt_lang=target_lang,
                    num_beams=3,
                    max_new_tokens=8000,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    length_penalty=2.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Process model output
            if isinstance(outputs, tuple):
                audio_output = outputs[0].cpu().numpy()
            else:
                audio_output = outputs.cpu().numpy()
            
            # Convert numpy array to torch tensor
            output_tensor = torch.from_numpy(audio_output).unsqueeze(0)
            
            return {
                "audio": output_tensor,
                "transcripts": {
                    "source": source_text,
                    "target": target_text
                }
            }
            
        except Exception as e:
            logger.error(f"Seamless translation failed: {str(e)}")
            raise
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported"""
        return language_code in self.languages
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        return self.languages