# services/translation_strategy.py
from typing import Dict, Any, Optional 
import logging
import torch
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TranslationStrategy: 
    def __init__(self):
        self.logger = logging.getLogger(__name__)
     
    def select_strategy(self, audio_analysis: Dict) -> Dict:
        try:
            background_music = audio_analysis.get('background_music', {})
            music_confidence = background_music.get('music_confidence', 0.0)
            has_music = background_music.get('has_background_music', False)
            if has_music or music_confidence > 0.15: content_type = "speech_with_music"
            else: content_type = "speech_only"
            self.logger.info(f"Selected Content Type: {content_type}")
            return {'content_type': content_type, 'heard_characteristics': {'music': {'detected': has_music, 'confidence': music_confidence}}}
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}"); return {'content_type': 'speech_only'}

class TranslationBackend(ABC):
    @abstractmethod
    def initialize(self): pass
    
    @abstractmethod
    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str, target_lang: str) -> Dict[str, Any]: pass
    
    @abstractmethod
    def is_language_supported(self, language_code: str) -> bool: pass
    
    @abstractmethod
    def get_supported_languages(self) -> Dict[str, str]: pass

class TranslationManager:
    def __init__(self):
        self.backends: Dict[str, TranslationBackend] = {}
        self.default_backend: Optional[str] = None 
        logger.info("TranslationManager initialized.")
    
    def register_backend(self, name: str, backend: TranslationBackend, is_default: bool = False):
        logger.info(f"Registering backend: '{name}', Default: {is_default}")
        if not isinstance(backend, TranslationBackend):
            raise TypeError("Backend must be TranslationBackend instance")
        self.backends[name] = backend
        logger.info(f"Successfully registered backend: '{name}'.")
        if is_default or self.default_backend is None:
            self.default_backend = name; logger.info(f"'{name}' is default backend.")
    
    def get_backend(self, name: Optional[str] = None) -> TranslationBackend:
        backend_to_return = None
        target_backend_name = name or self.default_backend
        
        if target_backend_name and target_backend_name in self.backends:
            logger.debug(f"Retrieving backend: '{target_backend_name}'")
            backend_to_return = self.backends[target_backend_name]
        elif self.backends: 
            first_available = list(self.backends.keys())[0]
            logger.warning(f"Requested/default backend '{target_backend_name}' not found. Falling back to first available: '{first_available}'.")
            backend_to_return = self.backends[first_available]
        
        if backend_to_return is None:
            logger.critical("No translation backends registered/available.")
            raise ValueError("No suitable translation backend available")
        
        # Ensure backend is initialized (idempotent if already initialized)
        if hasattr(backend_to_return, 'initialize') and callable(backend_to_return.initialize) and \
           hasattr(backend_to_return, 'initialized') and not backend_to_return.initialized:
            try:
                logger.info(f"Backend '{target_backend_name}' requires initialization. Initializing now...")
                backend_to_return.initialize()
            except Exception as e_init:
                logger.error(f"Failed to initialize backend '{target_backend_name}': {e_init}", exc_info=True)
                # Depending on app design, might re-raise or allow app to continue with uninitialized backend
        return backend_to_return

    def get_available_backends(self) -> Dict[str, TranslationBackend]: return self.backends
    
    def select_backend_for_language(self, target_lang: str) -> TranslationBackend: 
        logger.debug(f"Selecting backend for target language (app_code): '{target_lang}'")
        try:
            default_b = self.get_backend() # Gets default, initializes if needed
            if default_b.is_language_supported(target_lang):
                logger.info(f"Default backend '{self.default_backend}' supports '{target_lang}'.")
                return default_b
        except ValueError: # No default backend at all
            logger.critical("No default backend in TranslationManager for language selection.")
            raise

        for name, backend_b in self.backends.items():
            if name == self.default_backend: continue 
            if hasattr(backend_b, 'initialize') and callable(backend_b.initialize) and \
               hasattr(backend_b, 'initialized') and not backend_b.initialized:
                try: backend_b.initialize()
                except: pass # Logged by get_backend if it was called
            if backend_b.is_language_supported(target_lang):
                logger.info(f"Alternative backend '{name}' supports '{target_lang}'.")
                return backend_b
        
        logger.warning(f"No backend explicitly supports '{target_lang}'. Falling back to default: '{self.default_backend}'.")
        return self.get_backend()