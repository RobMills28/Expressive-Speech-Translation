# services/model_manager.py
import os
import gc
import time
import torch
import logging
import threading
from transformers import (
    SeamlessM4TProcessor,
    SeamlessM4Tv2Model,
    SeamlessM4Tv2ForSpeechToText
)

logger = logging.getLogger(__name__)

# Module-level DEVICE constant, for this manager IF it loads models
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEAMLESS_MODEL_ENV_VAR = "SEAMLESS_MODEL_NAME_FOR_MM" # Specific env var name
DEFAULT_SEAMLESS_MODEL_FOR_MM = "facebook/seamless-m4t-v2-large"

class ModelManager:
    """
    Singleton class to manage OPTIONAL SeamlessM4T components,
    if any specific route explicitly requires them.
    Primary S2ST should use TranslationManager with CascadedBackend.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models_loaded = False # Different flag
                    cls._instance.DEVICE_INSTANCE = DEVICE 
                    cls._instance.processor = None
                    cls._instance.model = None
                    cls._instance.text_model = None
                    cls._instance.is_loading_seamless = False
                    logger.info("ModelManager (for optional SeamlessM4T) instance created. Models NOT loaded.")
        return cls._instance
    
    def _load_seamless_models_if_needed(self):
        if self._models_loaded and self.processor and self.model and self.text_model:
            return # Already loaded

        if self.is_loading_seamless:
            logger.info("ModelManager: SeamlessM4T models are already being loaded by another call.")
            return # Another thread is loading

        with ModelManager._lock: # Ensure only one thread loads
            if self._models_loaded and self.processor: # Check again after acquiring lock
                return

            self.is_loading_seamless = True
            model_name_to_load = os.getenv(SEAMLESS_MODEL_ENV_VAR, DEFAULT_SEAMLESS_MODEL_FOR_MM)
            logger.info(f"ModelManager: Explicit request to load SeamlessM4T model: {model_name_to_load} onto {self.DEVICE_INSTANCE}")
            
            try:
                auth_token = os.getenv('HUGGINGFACE_TOKEN')
                if not auth_token: logger.warning("ModelManager: HUGGINGFACE_TOKEN not found for SeamlessM4T.")
                
                self.processor = SeamlessM4TProcessor.from_pretrained(model_name_to_load, token=auth_token, trust_remote_code=True)
                self.model = SeamlessM4Tv2Model.from_pretrained(model_name_to_load, token=auth_token, torch_dtype=torch.float32)
                self.text_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name_to_load, token=auth_token, torch_dtype=torch.float32, use_safetensors=True)

                if self.DEVICE_INSTANCE.type == 'cuda' and torch.cuda.is_available():
                    self.model = self.model.to(self.DEVICE_INSTANCE)
                    self.text_model = self.text_model.to(self.DEVICE_INSTANCE)
                self.model.eval(); self.text_model.eval()
                
                self._models_loaded = True # Set flag after successful load
                logger.info(f"ModelManager: SeamlessM4T model '{model_name_to_load}' loaded successfully.")
            except Exception as e:
                logger.error(f"ModelManager: Error loading SeamlessM4T model '{model_name_to_load}': {e}", exc_info=True)
                self.processor = self.model = self.text_model = None
                self._models_loaded = False # Explicitly false on error
            finally:
                self.is_loading_seamless = False

    def get_model_components(self):
        """
        Get SeamlessM4T model components. Loads them only if called and not already loaded.
        This should ONLY be called by routes that specifically need SeamlessM4T.
        """
        logger.info("ModelManager: get_model_components (for SeamlessM4T) called.")
        self._load_seamless_models_if_needed()
        
        if not self._models_loaded or not self.processor:
             logger.error("ModelManager: SeamlessM4T components failed to load or were not loaded. Returning None.")
             return None, None, None, None 
        
        return self.processor, self.model, self.text_model, self.processor.tokenizer 

    def cleanup(self):
        """Clean up SeamlessM4T model resources, if they were loaded."""
        if not self._models_loaded and not self.processor:
            logger.info("ModelManager: SeamlessM4T models were not loaded, no cleanup needed from this manager.")
            return

        logger.info("ModelManager: Initiating cleanup of any loaded SeamlessM4T models...")
        try:
            if self.DEVICE_INSTANCE.type == 'cuda' and torch.cuda.is_available():
                if self.model: self.model.cpu()
                if self.text_model: self.text_model.cpu()
            
            del self.processor; del self.model; del self.text_model
            self.processor = None; self.model = None; self.text_model = None
            self._models_loaded = False
            
            gc.collect()
            if self.DEVICE_INSTANCE.type == 'cuda' and torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info("ModelManager: SeamlessM4T model cleanup completed.")
        except Exception as e:
            logger.error(f"ModelManager: Error during SeamlessM4T model cleanup: {e}", exc_info=True)