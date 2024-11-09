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
from typing import Optional, Dict, Any, Tuple, List

# Third-party imports
import torch
import psutil
import numpy as np
import torchaudio
from transformers import (
    SeamlessM4TModel, 
    SeamlessM4TProcessor, 
    SeamlessM4TTokenizer
)

# Flask imports
from flask import (
    Flask, 
    request, 
    jsonify, 
    send_file, 
    Response, 
    current_app
)
from flask_cors import CORS

# Logging imports
import logging
from logging.handlers import (
    RotatingFileHandler, 
    TimedRotatingFileHandler
)

# Local imports
from translate_speech import translate_audio as process_audio  # Give it an alias to avoid conflict

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
MAX_AUDIO_LENGTH = 60  # seconds
MAX_DURATION = 60 # seconds
SAMPLE_RATE = 16000
CHUNK_SIZE = 8192
MAX_RETRIES = 3
TIMEOUT = 300  # seconds
MEMORY_THRESHOLD = 0.9  # 90% memory usage threshold
BATCH_SIZE = 1
START_TIME = time.time()

# Configure logging
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Main application log
    main_handler = TimedRotatingFileHandler(
        log_dir / 'app.log',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    main_handler.setFormatter(detailed_formatter)
    
    # Error log
    error_handler = RotatingFileHandler(
        log_dir / 'error.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Performance log
    perf_handler = RotatingFileHandler(
        log_dir / 'performance.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    perf_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[main_handler, error_handler, perf_handler]
    )
    
    # Also log to console for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Disable other loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Signal Handlers 
def sigterm_handler(signum, frame):
    """Handle SIGTERM signal"""
    logger.info("Received SIGTERM signal")
    sys.exit(0)

def sigint_handler(signum, frame):
    """Handle SIGINT signal (Ctrl+C)"""
    logger.info("Received SIGINT signal")
    sys.exit(0)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/translate": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Accept",
            "Authorization",
            "X-Requested-With"
        ],
        "expose_headers": [
            "Content-Type", 
            "Content-Length", 
            "Accept-Ranges", 
            "Content-Disposition",
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods",
            "Access-Control-Allow-Headers",
            "Access-Control-Expose-Headers"
        ],
        "supports_credentials": False,
        "max_age": 120,  # Cache preflight requests
        "vary": "Origin"  # Important for browser caching
    }
})

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
    
    def _load_model(self):
        try:
            logger.info(f"Loading SeamlessM4T model: {MODEL_NAME}")
            self.is_initializing = True
            
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            self.processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME, token=auth_token)
            self.model = SeamlessM4TModel.from_pretrained(MODEL_NAME, token=auth_token)
            self.tokenizer = SeamlessM4TTokenizer.from_pretrained(MODEL_NAME, token=auth_token)
            
            if DEVICE.type == 'cuda':
                self.model = self.model.to(DEVICE)
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Model loaded on CPU")
            
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
        self.last_used = time.time()
        return self.processor, self.model, self.tokenizer
    
    def cleanup(self):
        if DEVICE.type == 'cuda':
            try:
                if self.model is not None:
                    self.model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"Error during model cleanup: {str(e)}")

class AudioProcessor:
    @staticmethod
    def validate_audio_length(audio_path: str) -> bool:
        try:
            metadata = torchaudio.info(audio_path)
            duration = metadata.num_frames / metadata.sample_rate
            logger.info(f"Audio duration: {duration:.2f} seconds")
            return duration <= MAX_AUDIO_LENGTH
        except Exception as e:
            logger.error(f"Error validating audio length: {str(e)}")
            return False

    @staticmethod
    def process_audio(audio_path: str) -> torch.Tensor:
        try:
            logger.info(f"Loading audio from: {audio_path}")
            # Get audio file info
            info = torchaudio.info(audio_path)
            logger.info(f"Audio info - Sample rate: {info.sample_rate}, Channels: {info.num_channels}")
            
            # Load audio
            audio, orig_freq = torchaudio.load(audio_path)
            logger.info(f"Original audio shape: {audio.shape}, Frequency: {orig_freq}Hz")
            
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
@require_model
@performance_logger
def translate_audio_endpoint():
    if request.method == 'OPTIONS':
        return '', 204
    
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    start_time = time.time()
    temp_files = []
    
    try:
        logger.info(f"Starting translation request {request_id}")
        ResourceMonitor.check_memory()
        
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400
        
        target_language = request.form.get('target_language')
        if not target_language:
            return jsonify({'error': 'No target language specified'}), 400

        # Validate language
        model_language = LANGUAGE_MAP.get(target_language, target_language)
        if model_language not in LANGUAGE_CODES:
            return jsonify({'error': f'Unsupported language: {target_language}'}), 400

        # Save and process audio file
        file_extension = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_input:
            temp_files.append(temp_input.name)
            file.save(temp_input.name)
            logger.info(f"Saved input file: {temp_input.name}")
            
            # Validate audio file
            metadata = torchaudio.info(temp_input.name)
            duration = metadata.num_frames / metadata.sample_rate
            
            if duration > MAX_DURATION:
                return jsonify({'error': f'Audio file too long (max {MAX_DURATION} seconds)'}), 400
            
            if not AudioProcessor.validate_audio_length(temp_input.name):
                return jsonify({'error': 'Invalid audio file'}), 400
            
            # Get model components and process audio
            model_manager = ModelManager()
            processor, model, tokenizer = model_manager.get_model_components()

            try:
                audio = AudioProcessor.process_audio(temp_input.name)
                audio_numpy = audio.squeeze().numpy()
                logger.info(f"Audio shape after processing: {audio_numpy.shape}")
                logger.info(f"Audio min/max values: {audio_numpy.min():.3f}/{audio_numpy.max():.3f}")
            except Exception as e:
                logger.error(f"Audio processing error: {str(e)}")
                return jsonify({'error': f'Failed to process audio file: {str(e)}'}), 400
            finally:
                logger.info("Audio processing try block completed")

            # Prepare model inputs
            try:
                logger.info("Preparing model inputs...")
                inputs = processor(
                    audios=audio_numpy,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    src_lang="eng",
                    tgt_lang=model_language
                )
                logger.info(f"Input keys available: {inputs.keys()}")
                
                if DEVICE.type == 'cuda':
                    inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}
            except Exception as e:
                logger.error(f"Input processing error: {str(e)}")
                return jsonify({'error': f'Failed to prepare audio for translation: {str(e)}'}), 500

# Generate translation
            try:
                logger.info(f"Starting translation to {model_language}")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        tgt_lang=model_language,
                        num_beams=5,
                        max_new_tokens=200,
                        use_cache=True
                    )
                logger.info("Translation generation completed")

                # Process the output
                audio_output = None
                if hasattr(outputs, 'waveform'):
                    audio_output = outputs.waveform[0].cpu().numpy()
                    logger.info("Processed output using waveform attribute")
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    audio_output = outputs[0].squeeze(0).cpu().numpy()
                    logger.info("Processed output using tuple format")
                else:
                    raise ValueError("Unexpected output format from model")
                
                if audio_output is None:
                    raise ValueError("No audio data generated")
                
                logger.info(f"Audio output shape: {audio_output.shape}")
            except Exception as e:
                logger.error(f"Translation generation failed: {str(e)}")
                raise ValueError(f"Failed to generate translation: {str(e)}")
                
            # Save and send the translated audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_files.append(temp_output.name)
                waveform_tensor = torch.tensor(audio_output).unsqueeze(0)
                torchaudio.save(
                    temp_output.name,
                    waveform_tensor,
                    sample_rate=16000  # SeamlessM4T uses 16kHz
                )
                
                with open(temp_output.name, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                # Add debug logging
                logger.info(f"Audio data length: {len(audio_data)}, First few bytes: {audio_data[:20]}")
                
                response = Response(
                    audio_data,
                    status=200,  # Force 200 OK instead of 206 Partial Content
                    mimetype='audio/wav',
                    headers={
                        'Content-Type': 'audio/wav',
                        'Content-Length': str(len(audio_data)),
                        'Accept-Ranges': 'bytes',
                        'Access-Control-Allow-Origin': 'http://localhost:3000',
                        'Access-Control-Allow-Methods': 'POST, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type',
                        'Access-Control-Expose-Headers': 'Content-Length',
                        'Cache-Control': 'no-cache',
                        'Content-Disposition': f'attachment; filename=translated_{Path(file.filename).stem}.wav'
                    }
                )
                
                # Add more debug logging
                logger.info(f"Sending response with {len(audio_data)} bytes of audio data")
                return response

    except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return jsonify({'error': f'Translation failed: {str(e)}'}), 500

    except Exception as e:
        return ErrorHandler.handle_error(e)
    
    finally:
        for temp_file in temp_files:
            cleanup_file(temp_file)
        logger.info(f"Request {request_id} completed in {time.time() - start_time:.2f}s")
        ResourceMonitor.check_memory()
        if DEVICE.type == 'cuda':
            ResourceMonitor.check_gpu()

@app.route('/health', methods=['GET'])
@performance_logger
def health_check():
    """Health check endpoint with detailed system information"""
    try:
        # Check Hugging Face token
        token = os.getenv('HUGGINGFACE_TOKEN')
        token_status = bool(token)

        # Check model components
        model_manager = ModelManager()
        processor, model, tokenizer = model_manager.get_model_components()
        
        # Get GPU information if available
        gpu_info = None
        if DEVICE.type == 'cuda':
            try:
                gpu_properties = torch.cuda.get_device_properties(0)
                gpu_info = {
                    'name': gpu_properties.name,
                    'total_memory': f"{gpu_properties.total_memory/1024**2:.2f}MB",
                    'memory_allocated': f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
                    'memory_reserved': f"{torch.cuda.memory_reserved()/1024**2:.2f}MB",
                    'utilization': f"{torch.cuda.utilization()}%",
                    'device_count': torch.cuda.device_count()
                }
            except Exception as e:
                logger.error(f"Error getting GPU info: {str(e)}")
                gpu_info = {'error': str(e)}
        
        # Get system information
        system_info = {
            'cpu_usage': psutil.cpu_percent(interval=1),  # 1 second interval
            'memory': {
                'total': f"{psutil.virtual_memory().total/1024**3:.2f}GB",
                'available': f"{psutil.virtual_memory().available/1024**3:.2f}GB",
                'used_percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': f"{psutil.disk_usage('/').total/1024**3:.2f}GB",
                'free': f"{psutil.disk_usage('/').free/1024**3:.2f}GB",
                'used_percent': psutil.disk_usage('/').percent
            },
            'python_version': sys.version,
            'platform': platform.platform()
        }

        # Check CORS configuration
        cors_info = {
            'enabled': True,
            'allowed_origins': app.config.get('CORS_ALLOWED_ORIGINS', ['http://localhost:3000']),
            'allowed_methods': ['GET', 'POST', 'OPTIONS'],
            'max_age': 120
        }
        
        # Get temp directory status
        temp_dir = tempfile.gettempdir()
        temp_files = list(Path(temp_dir).glob("*.wav"))
        temp_info = {
            'path': temp_dir,
            'wav_file_count': len(temp_files),
            'total_size_mb': sum(f.stat().st_size for f in temp_files) / (1024 * 1024)
        }

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model': {
                'name': MODEL_NAME,
                'loaded': None not in (processor, model, tokenizer),
                'device': DEVICE.type,
                'last_used': model_manager.last_used,
                'huggingface_token_configured': token_status
            },
            'system': system_info,
            'gpu': gpu_info,
            'languages_supported': LANGUAGE_CODES,
            'cors': cors_info,
            'temp_files': temp_info,
            'uptime': time.time() - START_TIME  # Add START_TIME at the top of your file
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
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
        # Validate environment
        if not os.getenv('HUGGINGFACE_TOKEN'):
            logger.error("HUGGINGFACE_TOKEN not set in environment")
            sys.exit(1)

        # Check model loading
        model_manager = ModelManager()  # Create instance first
        if None in model_manager.get_model_components():
            logger.error("Could not start the app due to model loading failure")
            sys.exit(1)
        
        # Initial cleanup
        cleanup_temp_files()
        
        # Log startup information
        logger.info("="*50)  # Add visual separator in logs
        logger.info("Starting LinguaSync Backend")
        logger.info(f"Starting Flask app on port 5001 using {DEVICE}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Supported languages: {list(LANGUAGE_CODES.keys())}")
        
        if DEVICE.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            ResourceMonitor.check_gpu()
            
        logger.info("="*50)  # Add visual separator in logs
        
        # Start the application
        app.run(
            debug=False,  # Set to False for production
            host='0.0.0.0',
            port=5001,
            use_reloader=False,  # Prevent duplicate model loading
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        sys.exit(1)