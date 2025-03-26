from typing import Dict, Any
import logging
import torch
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TranslationStrategy:
    """Selects appropriate translation strategy based on audio characteristics."""
     
    def __init__(self):
        self.logger = logging.getLogger(__name__)
     
    def select_strategy(self, audio_analysis: Dict) -> Dict:
        """Select translation strategy based on audio characteristics"""
        try:
            # Get basic characteristics
            background_music = audio_analysis.get('background_music', {})
            music_confidence = background_music.get('music_confidence', 0.0)
            has_music = background_music.get('has_background_music', False)
             
            # Simple content type determination
            if has_music or music_confidence > 0.15:
                content_type = "speech_with_music"
                self.logger.info(f"Content Type: speech_with_music (Music confidence: {music_confidence})")
            else:
                content_type = "speech_only"
                self.logger.info(f"Content Type: speech_only (Music confidence: {music_confidence})")
             
            # Return basic strategy
            return {
                'content_type': content_type,
                'heard_characteristics': {
                    'music': {
                        'detected': has_music,
                        'confidence': music_confidence
                    }
                }
            }
             
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {str(e)}")
            return {'content_type': 'speech_only'}  # Safe default

class TranslationBackend(ABC):
    """Abstract base class for translation backends"""
    
    @abstractmethod
    def initialize(self):
        """Initialize models and resources"""
        pass
    
    @abstractmethod
    def translate_speech(
        self, 
        audio_tensor: torch.Tensor, 
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Translate speech from source to target language
        
        Returns:
            Dictionary with translated audio, source text, and target text
        """
        pass
    
    @abstractmethod
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language is supported by this backend"""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        pass

class TranslationManager:
    """Manages multiple translation backends"""
    
    def __init__(self):
        self.backends = {}
        self.default_backend = None
    
    def register_backend(self, name: str, backend: TranslationBackend, is_default: bool = False):
        """Register a translation backend"""
        self.backends[name] = backend
        if is_default:
            self.default_backend = name
    
    def get_backend(self, name: str = None) -> TranslationBackend:
        """Get a specific backend or the default one"""
        if name and name in self.backends:
            return self.backends[name]
        elif self.default_backend:
            return self.backends[self.default_backend]
        else:
            raise ValueError("No translation backend available")
    
    # Add this method to fix the first error
    def get_available_backends(self) -> dict:
        """Get all available backends"""
        return self.backends
    
    def select_backend_for_language(self, target_lang: str) -> TranslationBackend:
        """Select appropriate backend for a given language"""
        # Try default backend first
        if self.default_backend and self.backends[self.default_backend].is_language_supported(target_lang):
            return self.backends[self.default_backend]
        
        # Try other backends
        for name, backend in self.backends.items():
            if backend.is_language_supported(target_lang):
                return backend
        
        raise ValueError(f"No backend supports language: {target_lang}")