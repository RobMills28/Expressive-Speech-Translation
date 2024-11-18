# Standard library imports
import os
import sys
import gc
import json
import time
import atexit
import signal
import shutil
import hashlib
import warnings
import tempfile
import threading
import traceback
import platform
import queue
from datetime import datetime
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, List, Callable

# Third-party imports: Monitoring and Metrics
from prometheus_client import Counter, Histogram, start_http_server
import psutil

# Third-party imports: ML and Audio
import torch
import numpy as np
import torchaudio
import scipy.signal
from transformers import (
    SeamlessM4TModel, 
    SeamlessM4TProcessor, 
    SeamlessM4TTokenizer
)

# Flask and Extensions
from flask import (
    Flask, 
    request, 
    jsonify,
    make_response, 
    send_file, 
    Response, 
    current_app
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Logging
import logging
from logging.handlers import (
    RotatingFileHandler, 
    TimedRotatingFileHandler
)

# Environment variables
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
MAX_AUDIO_LENGTH = 300  # seconds (5 minutes)
MAX_DURATION = 300 # seconds
MAX_TOKENS = 1500
SAMPLE_RATE = 16000
CHUNK_SIZE = 16384  # Increased for better processing
MAX_RETRIES = 3
TIMEOUT = 600  # seconds (increased for longer audio)
MEMORY_THRESHOLD = 0.9
BATCH_SIZE = 1
START_TIME = time.time()

# Initialize Flask app first
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Accept",
            "Authorization",
            "X-Requested-With",
            "Range",
            "Accept-Ranges",
            "Origin"
        ],
        "expose_headers": [
            "Content-Type", 
            "Content-Length", 
            "Content-Range",
            "Content-Disposition",
            "Accept-Ranges",
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Credentials"
        ],
        "supports_credentials": True,
        "max_age": 120,
        "automatic_options": True
    }
})

# Metrics configuration
TRANSLATION_REQUESTS = Counter('translation_requests_total', 'Total translation requests')
TRANSLATION_ERRORS = Counter('translation_errors_total', 'Total translation errors')
PROCESSING_TIME = Histogram('translation_processing_seconds', 'Time spent processing translations')

# Initialize rate limiter after Flask app
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure logging
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    main_handler = TimedRotatingFileHandler(
        log_dir / 'app.log',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    main_handler.setFormatter(detailed_formatter)
    
    error_handler = RotatingFileHandler(
        log_dir / 'error.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    perf_handler = RotatingFileHandler(
        log_dir / 'performance.log',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    perf_handler.setFormatter(detailed_formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[main_handler, error_handler, perf_handler]
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)
    logging.getLogger().addHandler(console_handler)
    
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def graceful_shutdown():
    """Graceful shutdown handler"""
    try:
        logger.info("Initiating graceful shutdown...")
        app.config['IS_SHUTTING_DOWN'] = True
        time.sleep(5)
        shutdown_handler()
        logger.info("Graceful shutdown completed")
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {str(e)}")
    finally:
        sys.exit(0)

def sigterm_handler(signum, frame):
    logger.info("Received SIGTERM signal")
    graceful_shutdown()

def sigint_handler(signum, frame):
    logger.info("Received SIGINT signal")
    graceful_shutdown()

# Request Validation Middleware
@app.before_request
def validate_request():
    if request.method not in ['GET', 'POST', 'OPTIONS']:
        return jsonify({'error': 'Method not allowed'}), 405
    if request.method == 'POST' and not request.is_json and not request.files:
        return jsonify({'error': 'Invalid content type'}), 400

# Request Timer Middleware
@app.before_request
def start_timer():
    request.start_time = time.time()

# Handle OPTIONS preflight requests
@app.route('/translate', methods=['OPTIONS'])
def handle_preflight():
    response = make_response()
    origin = request.headers.get('Origin', 'http://localhost:3000')
    
    response.headers.update({
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Accept, Authorization, Range, Accept-Ranges, Origin',
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Max-Age': '120',
        'Content-Type': 'text/plain'
    })
    return response

# Response Handlers
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin', 'http://localhost:3000')
    
    # Basic CORS headers
    response.headers.update({
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Accept, Authorization, Range, Accept-Ranges, Origin',
        'Access-Control-Expose-Headers': 'Content-Range, Content-Length, Accept-Ranges, Access-Control-Allow-Origin',
        'Access-Control-Max-Age': '120',
        'Vary': 'Origin'
    })
    
    # Additional headers for audio responses
    if response.mimetype == 'audio/wav':
        response.headers.update({
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Content-Type': 'audio/wav'
        })
    
    # Log request duration
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"Request to {request.path} took {duration:.2f}s")
    
    return response

# Error handler for server errors
@app.errorhandler(500)
def handle_error(e):
    if request.method == 'OPTIONS':
        response = make_response()
        origin = request.headers.get('Origin', 'http://localhost:3000')
        
        response.headers.update({
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Accept, Authorization, Range, Accept-Ranges, Origin',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Max-Age': '120',
            'Content-Type': 'text/plain'
        })
        return response, 200
    
    return jsonify(error=str(e)), 500

# Model configuration
MODEL_NAME = "facebook/seamless-m4t-v2-large"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auth_token = os.getenv('HUGGINGFACE_TOKEN')

if not auth_token:
    logger.error("HUGGINGFACE_TOKEN not found in environment variables")
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")

# Language configurations
LANGUAGE_CODES = {
    'fra': ('fra', 'French'),
    'spa': ('spa', 'Spanish'),
    'deu': ('deu', 'German'),
    'ita': ('ita', 'Italian'),
    'por': ('por', 'Portuguese')
}

LANGUAGE_MAP = {
    'de': 'deu',
    'fr': 'fra',
    'es': 'spa',
    'it': 'ita',
    'pt': 'por'
}

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.last_used = None
        self.is_initializing = False
        self._load_model()

    def _verify_model(self):
        """Verify model loaded correctly"""
        try:
            # Create proper dummy inputs for verification
            sample_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
            
            # Process through the processor correctly
            inputs = self.processor(
                audios=sample_audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                src_lang="eng",  # Source language English
                tgt_lang="fra"   # Target language French for testing
            )
            
            # Move to device if using CUDA
            if DEVICE.type == 'cuda':
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Try generating output using generate() instead of forward()
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

    def _load_model(self):
        try:
            logger.info(f"Loading SeamlessM4T model: {MODEL_NAME}")
            self.is_initializing = True
            
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load model components with timeout and performance tracking
            with PROCESSING_TIME.time():
                self.processor = SeamlessM4TProcessor.from_pretrained(
                    MODEL_NAME, 
                    token=auth_token,
                    cache_dir="./model_cache"
                )
                self.model = SeamlessM4TModel.from_pretrained(
                    MODEL_NAME, 
                    token=auth_token,
                    cache_dir="./model_cache"
                )
                self.tokenizer = SeamlessM4TTokenizer.from_pretrained(
                    MODEL_NAME, 
                    token=auth_token,
                    cache_dir="./model_cache"
                )
            
            if DEVICE.type == 'cuda':
                self.model = self.model.to(DEVICE)
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
                # Log GPU memory usage after loading
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
        """Get model components with validation and monitoring"""
        try:
            # Check if model needs reloading (e.g., if it's been too long since last use)
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

# Replace your existing AudioProcessor class
class AudioProcessor:
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac'}

    @staticmethod
    def validate_audio_length(audio_path: str) -> tuple[bool, str]:
        """
        Validates audio file length and basic integrity
        Returns: (is_valid: bool, error_message: str)
        """
        try:
            # Format validation
            if Path(audio_path).suffix.lower() not in AudioProcessor.SUPPORTED_FORMATS:
                return False, f"Unsupported audio format. Supported: {AudioProcessor.SUPPORTED_FORMATS}"

            # File checks
            if not os.path.exists(audio_path):
                return False, "Audio file not found"
                
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                return False, "Audio file is empty"

            # Get audio metadata
            metadata = torchaudio.info(audio_path)
            
            # Check sample rate
            if metadata.sample_rate <= 0:
                return False, "Invalid sample rate detected"
                
            # Check number of frames
            if metadata.num_frames <= 0:
                return False, "No audio frames detected"

            # Calculate duration
            duration = metadata.num_frames / metadata.sample_rate
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            if duration <= 0:
                return False, "Invalid audio duration"
                
            if duration > MAX_AUDIO_LENGTH:
                return False, f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({MAX_AUDIO_LENGTH}s)"

            return True, ""

        except Exception as e:
            error_msg = f"Error validating audio: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def process_audio(audio_path: str) -> torch.Tensor:
        try:
            logger.info(f"Loading audio from: {audio_path}")
            
            # Load and check audio
            info = torchaudio.info(audio_path)
            logger.info(f"Audio info - Sample rate: {info.sample_rate}, Channels: {info.num_channels}")
            
            audio, orig_freq = torchaudio.load(audio_path)
            
            # Quality checks
            if torch.isnan(audio).any():
                raise ValueError("Audio contains NaN values")
            if torch.isinf(audio).any():
                raise ValueError("Audio contains infinite values")
            if audio.abs().max() == 0:
                raise ValueError("Audio is silent")
            
            logger.info(f"Original audio shape: {audio.shape}, Frequency: {orig_freq}Hz")
            
            # Process in chunks if large
            if audio.shape[1] > 1_000_000:  # If longer than 1M samples
                chunks = audio.split(1_000_000, dim=1)
                audio = torch.cat([chunk for chunk in chunks], dim=1)
            
            # Resample if needed
            if orig_freq != SAMPLE_RATE:
                logger.info(f"Resampling from {orig_freq}Hz to {SAMPLE_RATE}Hz")
                audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=SAMPLE_RATE)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                logger.info("Converting from stereo to mono")
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Normalize audio
            if audio.abs().max() > 1.0:
                logger.info("Normalizing audio")
                audio = audio / audio.abs().max()
            
            logger.info(f"Processed audio shape: {audio.shape}")
            return audio
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process audio: {str(e)}")

class ResourceMonitor:
    @staticmethod
    def check_memory():
        memory = psutil.virtual_memory()
        if memory.percent > MEMORY_THRESHOLD * 100:
            logger.warning(f"High memory usage: {memory.percent}%")
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    @staticmethod
    def check_gpu():
        if DEVICE.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            logger.info(f"GPU Memory - Allocated: {memory_allocated/1024**2:.2f}MB, Reserved: {memory_reserved/1024**2:.2f}MB")

    @staticmethod
    def check_resources() -> tuple[bool, str]:
        """Check system resources before processing"""
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > MEMORY_THRESHOLD * 100:
                return False, f"System memory usage too high ({memory.percent}%)"

            # Check GPU memory if available
            if DEVICE.type == 'cuda':
                gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
                if gpu_memory_used > 0.9:
                    return False, f"GPU memory usage too high ({gpu_memory_used*100:.1f}%)"

            return True, ""
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            return False, "Resource check failed"

class ErrorHandler:
    @staticmethod
    def handle_error(e: Exception) -> Tuple[dict, int]:
        error_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        error_info = {
            'error_id': error_id,
            'type': type(e).__name__,
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(f"Error {error_id}: {str(e)}", exc_info=True)
        
        user_message = f"An error occurred (ID: {error_id}). Please try again or contact support if the problem persists."
        return {'error': user_message}, 500

def generate_progress_event(progress: int, status: str) -> str:
    return f"data: {json.dumps({'progress': progress, 'status': status})}\n\n"

def cleanup_file(file_path: str) -> None:
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

def require_model(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        model_manager = ModelManager()
        processor, model, tokenizer = model_manager.get_model_components()
        
        if None in (processor, model, tokenizer):
            return jsonify({'error': 'Model not initialized properly'}), 503
            
        return f(*args, **kwargs)
    return decorated_function

def performance_logger(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = f(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            duration = end_time - start_time
            memory_diff = end_memory - start_memory
            
            logger.info(f"Performance - Function: {f.__name__}, Duration: {duration:.2f}s, "
                       f"Memory Change: {memory_diff/1024/1024:.2f}MB")
            
            return result
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            raise
            
    return decorated_function

@app.route('/translate', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
@require_model
@performance_logger
def translate_audio_endpoint():
    # Initialize metrics
    TRANSLATION_REQUESTS.inc()
    
    if request.method == 'OPTIONS':
        return '', 204
    
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    start_time = time.time()
    temp_files = []
    
    try:
        logger.info(f"Starting translation request {request_id}")
        
        # Resource check
        resources_ok, resource_error = ResourceMonitor.check_resources()
        if not resources_ok:
            raise ValueError(f"Resource check failed: {resource_error}")
        
        # Validate request
        if 'file' not in request.files:
            logger.error(f"Request {request_id}: No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename:
            logger.error(f"Request {request_id}: Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        target_language = request.form.get('target_language')
        if not target_language:
            logger.error(f"Request {request_id}: No target language specified")
            return jsonify({'error': 'No target language specified'}), 400

        # Validate language
        model_language = LANGUAGE_MAP.get(target_language, target_language)
        if model_language not in LANGUAGE_CODES:
            logger.error(f"Request {request_id}: Unsupported language {target_language}")
            return jsonify({'error': f'Unsupported language: {target_language}'}), 400

        # Save and process audio file with proper error handling
        file_extension = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_input:
            temp_files.append(temp_input.name)
            file.save(temp_input.name)
            
            # Validate file was actually saved
            if not os.path.exists(temp_input.name) or os.path.getsize(temp_input.name) == 0:
                raise ValueError("Failed to save audio file")
            
            logger.info(f"Request {request_id}: Saved input file: {temp_input.name}")
            
            # Validate audio file
            is_valid, error_message = AudioProcessor.validate_audio_length(temp_input.name)
            if not is_valid:
                logger.error(f"Request {request_id}: Audio validation failed - {error_message}")
                return jsonify({'error': error_message}), 400
            
            # Process audio with enhanced error checking
            try:
                audio = AudioProcessor.process_audio(temp_input.name)
                audio_numpy = audio.squeeze().numpy()
                
                if np.isnan(audio_numpy).any() or np.isinf(audio_numpy).any():
                    raise ValueError("Invalid audio data detected")
                
                logger.info(f"Request {request_id}: Audio processed successfully")
            except Exception as e:
                logger.error(f"Audio processing error: {str(e)}")
                return jsonify({'error': f'Failed to process audio: {str(e)}'}), 400

            # Prepare model inputs with validation
            try:
                inputs = processor(
                    audios=audio_numpy,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    src_lang="eng",
                    tgt_lang=model_language
                )
                
                if not inputs or not inputs.keys():
                    raise ValueError("Failed to prepare model inputs")
                
                if DEVICE.type == 'cuda':
                    inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}
                
            except Exception as e:
                logger.error(f"Input processing error: {str(e)}")
                return jsonify({'error': f'Failed to prepare audio: {str(e)}'}), 400

            # Generate translation with enhanced error handling
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        tgt_lang=model_language,
                        num_beams=5,
                        max_new_tokens=500,  # Increased from 200
                        min_new_tokens=50,   # Added minimum
                        use_cache=True,
                        temperature=0.7,     # Added temperature
                        length_penalty=1.2,  # Increased from 1.0
                        no_repeat_ngram_size=3,  # Added to prevent repetition
                        early_stopping=True
                    )

                # Add output debugging
                logger.info(f"Model output shape: {outputs.shape if hasattr(outputs, 'shape') else 'No shape'}")
                logger.info(f"Model output type: {type(outputs)}")

                if outputs is None:
                    raise ValueError("Model generated no output")
                    
            except Exception as e:
                logger.error(f"Translation error: {str(e)}")
                return jsonify({'error': f'Translation failed: {str(e)}'}), 500

            # Process the output with validation
            try:
                if hasattr(outputs, 'waveform'):
                    audio_output = outputs.waveform[0].cpu().numpy()
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    audio_output = outputs[0].squeeze(0).cpu().numpy()
                else:
                    raise ValueError("Unexpected output format from model")

                if audio_output is None or audio_output.size == 0:
                    raise ValueError("No audio data generated")
                
                # Normalize audio if needed
                if np.abs(audio_output).max() > 1.0:
                    audio_output = audio_output / np.abs(audio_output).max()
                
                logger.info(f"Request {request_id}: Audio output processed successfully")
                
            except Exception as e:
                logger.error(f"Output processing error: {str(e)}")
                return jsonify({'error': f'Failed to process translation: {str(e)}'}), 500

            # Save and send the translated audio with proper headers
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                    temp_files.append(temp_output.name)
                    waveform_tensor = torch.tensor(audio_output).unsqueeze(0)
                    torchaudio.save(
                        temp_output.name,
                        waveform_tensor,
                        sample_rate=16000
                    )
                    
                    # Verify the saved file
                    if not os.path.exists(temp_output.name) or os.path.getsize(temp_output.name) == 0:
                        raise ValueError("Failed to save translated audio")
                    
                    with open(temp_output.name, 'rb') as audio_file:
                        audio_data = audio_file.read()
                    
                    if not audio_data:
                        raise ValueError("Generated audio data is empty")

                    response = Response(
                        audio_data,
                        mimetype='audio/wav',
                        headers={
                            'Content-Type': 'audio/wav',
                            'Content-Length': str(len(audio_data)),
                            'Content-Disposition': f'attachment; filename=translated_{Path(file.filename).stem}.wav',
                            'Cache-Control': 'no-cache, no-store, must-revalidate',
                            'Pragma': 'no-cache',
                            'Expires': '0',
                            'Access-Control-Allow-Origin': request.headers.get('Origin', 'http://localhost:3000'),
                            'Access-Control-Allow-Methods': 'POST, OPTIONS',
                            'Access-Control-Allow-Headers': 'Content-Type',
                            'Access-Control-Allow-Credentials': 'true'
                        }
                    )
                    
                    logger.info(f"Request {request_id}: Response prepared successfully")
                    return response

            except Exception as e:
                logger.error(f"Response preparation error: {str(e)}")
                return jsonify({'error': f'Failed to prepare response: {str(e)}'}), 500

    except Exception as e:
        TRANSLATION_ERRORS.inc()
        logger.error(f"Request {request_id}: Unhandled error: {str(e)}", exc_info=True)
        return ErrorHandler.handle_error(e)

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            cleanup_file(temp_file)
        
        duration = time.time() - start_time
        logger.info(f"Request {request_id} completed in {duration:.2f}s")
        PROCESSING_TIME.observe(duration)
        
        # Resource monitoring
        ResourceMonitor.check_memory()
        if DEVICE.type == 'cuda':
            ResourceMonitor.check_gpu()
    
@app.route('/health/model', methods=['GET'])
@performance_logger
def model_health():
    """Detailed model health check"""
    try:
        model_manager = ModelManager()
        
        health_info = {
            'status': 'healthy',
            'model_loaded': all(x is not None for x in model_manager.get_model_components()),
            'last_used': model_manager.last_used,
            'memory_usage': {
                'system': psutil.Process().memory_info().rss / 1024**2,
                'system_percent': psutil.virtual_memory().percent,
            }
        }
        
        if DEVICE.type == 'cuda':
            health_info['memory_usage']['gpu'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'reserved': torch.cuda.memory_reserved() / 1024**2,
            }
        
        return jsonify(health_info)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


def cleanup_temp_files():
    """Clean up any temporary files in the temp directory"""
    try:
        temp_dir = tempfile.gettempdir()
        pattern = Path(temp_dir) / "*.wav"
        files_removed = 0
        bytes_freed = 0
        
        for file in Path(temp_dir).glob("*.wav"):
            try:
                size = file.stat().st_size
                cleanup_file(str(file))
                files_removed += 1
                bytes_freed += size
                logger.debug(f"Removed temp file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file}: {str(e)}")
        
        logger.info(f"Cleanup complete: removed {files_removed} files, freed {bytes_freed/1024**2:.2f}MB")
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {str(e)}", exc_info=True)


def shutdown_handler():
    """Clean up resources during shutdown"""
    logger.info("Application shutting down...")
    try:
        # Release GPU memory if using CUDA
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Cleanup model resources
        model_manager = ModelManager()
        model_manager.cleanup()
        
        # Remove temporary files
        cleanup_temp_files()
        
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {str(e)}", exc_info=True)


# Add at the top of your file
START_TIME = time.time()

# Register shutdown handlers
atexit.register(shutdown_handler)
signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigint_handler)  # Changed to use sigint_handler

if __name__ == '__main__':
    try:
        # Start metrics server
        start_http_server(8000)
        logger.info("Metrics server started on port 8000")

        # Validate environment
        if not os.getenv('HUGGINGFACE_TOKEN'):
            logger.error("HUGGINGFACE_TOKEN not set in environment")
            sys.exit(1)

        # Check model loading
        model_manager = ModelManager()  
        processor, model, tokenizer = model_manager.get_model_components()
        if None in (processor, model, tokenizer):
            logger.error("Could not start the app due to model loading failure")
            sys.exit(1)
        
        # Initial resource check
        resources_ok, resource_error = ResourceMonitor.check_resources()
        if not resources_ok:
            logger.error(f"Resource check failed during startup: {resource_error}")
            sys.exit(1)
        
        # Initial cleanup
        cleanup_temp_files()
        
        # Log startup information
        logger.info("="*50)
        logger.info("Starting LinguaSync Backend")
        logger.info(f"Starting Flask app on port 5001 using {DEVICE}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Supported languages: {list(LANGUAGE_CODES.keys())}")
        logger.info(f"Maximum audio length: {MAX_AUDIO_LENGTH} seconds")
        logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
        
        # Log system information
        memory = psutil.virtual_memory()
        logger.info(f"System memory: {memory.total/1024**3:.1f}GB total, {memory.available/1024**3:.1f}GB available")
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        
        if DEVICE.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            ResourceMonitor.check_gpu()
            
        # Log configuration information
        logger.info("Configuration:")
        logger.info(f"- Debug mode: {app.debug}")
        logger.info(f"- CORS enabled: {bool(app.config.get('CORS_ENABLED', True))}")
        logger.info(f"- Rate limiting: 10 requests per minute")
        logger.info(f"- Metrics enabled: True (port 8000)")
        
        logger.info("="*50)
        
        # Initialize rate limiter storage
        limiter.init_app(app)
        
        # Register shutdown handlers
        atexit.register(shutdown_handler)
        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigint_handler)
        
        # Start the application with simplified configuration
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False,
            use_reloader=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        # Log full stack trace
        logger.error("Stack trace:", exc_info=True)
        # Try to clean up resources if possible
        try:
            if 'model_manager' in locals():
                model_manager.cleanup()
            cleanup_temp_files()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed during error handling: {str(cleanup_error)}")
        sys.exit(1)