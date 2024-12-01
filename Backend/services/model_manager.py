import os
import gc
import time
import torch
import logging
import threading
from transformers import (
    AutoProcessor,
    SeamlessM4Tv2Model,
    SeamlessM4TTokenizer
)

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "facebook/seamless-m4t-v2-large"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelManager:
    """
    Singleton class to manage the ML model lifecycle and resources.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize model manager state"""
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.last_used = None
        self.is_initializing = False
        self._load_model()

    def _verify_model(self) -> bool:
        """
        Verify model loaded correctly
        
        Returns:
            bool: True if model verification successful, False otherwise
        """
        try:
            # Create proper dummy inputs for verification
            sample_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
            
            # Process through the processor correctly
            inputs = self.processor(
                audios=sample_audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                src_lang="eng",
                tgt_lang="fra"
            )
            
            # Move to device if using CUDA
            if DEVICE.type == 'cuda':
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Try generating output using generate()
                outputs = self.model.generate(
                    **inputs,
                    tgt_lang="fra",
                    num_beams=1,
                    max_new_tokens=50
                )
                
            logger.info("Model verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            return False

    def _verify_model_weights(self):
        """Verify model weights are properly initialized."""
        uninitialized = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.data.mean().item() == 0:
                uninitialized.append(name)
        return uninitialized

    def _load_model(self):
        """Load model components with verification"""
        try:
            logger.info(f"Loading SeamlessM4T model: {MODEL_NAME}")
            self.is_initializing = True
            
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Get auth token
            auth_token = os.getenv('HUGGINGFACE_TOKEN')
            if not auth_token:
                raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
            
            # Load model components with improved initialization
            self.processor = AutoProcessor.from_pretrained(
                    MODEL_NAME, 
                    token=auth_token,
                    cache_dir="./model_cache",
                    trust_remote_code=True,
                    use_safetensors=True,
                    revision="main"
                )
                
            self.model = SeamlessM4Tv2Model.from_pretrained(
                    MODEL_NAME, 
                    token=auth_token,
                    cache_dir="./model_cache",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    use_safetensors=True,
                    revision="main"
                )
                
            self.tokenizer = SeamlessM4TTokenizer.from_pretrained(
                    MODEL_NAME, 
                    token=auth_token,
                    cache_dir="./model_cache",
                    trust_remote_code=True,
                    use_safetensors=True,
                    revision="main"
                )
            
            # Add weight verification
            uninitialized = self._verify_model_weights()
            if uninitialized:
                logger.warning(f"Found {len(uninitialized)} uninitialized weights: {uninitialized[:5]}...")
            
            if DEVICE.type == 'cuda':
                self.model = self.model.to(DEVICE)
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                logger.info(f"GPU Memory after loading - Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")
            else:
                logger.info("Model loaded on CPU")
            
            # Verify model immediately after loading
            if not self._verify_model():
                raise ValueError("Model verification failed after loading")
            
            self.last_used = time.time()
            logger.info("Model initialization successful")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}", exc_info=True)
            self.processor = None
            self.model = None
            self.tokenizer = None
            raise
        finally:
            self.is_initializing = False
    
    def get_model_components(self):
        """
        Get model components with validation and monitoring
        
        Returns:
            tuple: (processor, model, tokenizer) or (None, None, None) if error
        """
        try:
            # Check if model needs reloading
            if self.last_used and time.time() - self.last_used > 3600:  # 1 hour
                logger.info("Model inactive for too long, reloading...")
                self._load_model()
            
            # Verify model state
            if not self._verify_model():
                logger.error("Model verification failed during component request")
                return None, None, None
            
            # Update last used timestamp
            self.last_used = time.time()
            
            # Log memory usage
            if DEVICE.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                logger.debug(f"GPU Memory at component request: {memory_allocated:.2f}MB")
            
            return self.processor, self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error getting model components: {str(e)}")
            return None, None, None
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            logger.info("Cleaning up model resources")
            if DEVICE.type == 'cuda':
                if self.model is not None:
                    self.model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
                
            self.processor = None
            self.model = None
            self.tokenizer = None
            logger.info("Model cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}")
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()