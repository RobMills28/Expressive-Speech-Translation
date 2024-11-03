import os
from flask import Flask, request, jsonify, send_file, Response, current_app
from flask_cors import CORS 
from translate_speech import translate_audio as process_audio  # Give it an alias to avoid conflictfrom flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import torch
import torchaudio
import tempfile
from pathlib import Path
from transformers import SeamlessM4TModel, SeamlessM4TProcessor, SeamlessM4TTokenizer
import numpy as np
import json
import gc
from functools import wraps
import time
import sys
import signal
import atexit
import psutil
import warnings
from datetime import datetime
import traceback
from typing import Optional, Dict, Any, Tuple, List
import threading
import queue
import hashlib
import shutil

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

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/translate": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

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

setup_logging()
logger = logging.getLogger(__name__)

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
            
# Get model components first
            model_manager = ModelManager()
            processor, model, tokenizer = model_manager.get_model_components()

            # Then process audio
            try:
                audio = AudioProcessor.process_audio(temp_input.name)
                audio_numpy = audio.squeeze().numpy()
                
                logger.info(f"Audio shape after processing: {audio_numpy.shape}")
                logger.info(f"Audio min/max values: {audio_numpy.min():.3f}/{audio_numpy.max():.3f}")
                
            except Exception as e:
                logger.error(f"Audio processing error: {str(e)}")
                return jsonify({'error': f'Failed to process audio file: {str(e)}'}), 400

            # Prepare model inputs
            try:
                logger.info("Preparing model inputs...")
                inputs = processor(
                    audios=audio_numpy,  # Changed from 'audio' to 'audios'
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
                        **inputs,  # Pass all inputs directly
                        tgt_lang=model_language,
                        num_beams=5,
                        max_new_tokens=200,
                        use_cache=True
                    )
                logger.info("Translation generation completed")

                try:
                    # Process output
                    if hasattr(outputs, 'waveform'):
                        audio_output = outputs.waveform[0].cpu().numpy()
                        logger.info("Processed output using waveform attribute")
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        # Handle 3D tensor by taking the first element
                        audio_output = outputs[0].squeeze(0).cpu().numpy()  # Add squeeze here
                        logger.info("Processed output using tuple format")
                    else:
                        raise ValueError("Unexpected output format from model")
                    
                    logger.info(f"Audio output shape: {audio_output.shape}")
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                        temp_files.append(temp_output.name)
                        # Ensure audio is 2D for torchaudio.save
                        if len(audio_output.shape) == 1:
                            waveform_tensor = torch.tensor(audio_output).unsqueeze(0)
                        else:
                            waveform_tensor = torch.tensor(audio_output)
                            
                        torchaudio.save(
                            temp_output.name,
                            waveform_tensor,
                            sample_rate=SAMPLE_RATE
                        )

                        # Clean up GPU memory if needed
                        if DEVICE.type == 'cuda':
                            torch.cuda.empty_cache()
                            gc.collect()

                        duration = time.time() - start_time
                        logger.info(f"Translation {request_id} completed in {duration:.2f} seconds")
                        
                        return send_file(
                            temp_output.name,
                            mimetype='audio/wav',
                            as_attachment=True,
                            download_name=f"translated_{Path(file.filename).stem}.wav"
                        )

                except Exception as e:
                    logger.error(f"Output processing error: {str(e)}")
                    return jsonify({'error': f'Failed to process translated audio: {str(e)}'}), 500

            except Exception as e:
                logger.error(f"Translation error: {str(e)}")
                return jsonify({'error': f'Translation failed: {str(e)}'}), 500

    except Exception as e:
        return ErrorHandler.handle_error(e)
        
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            cleanup_file(temp_file)
        
        # Log memory usage
        ResourceMonitor.check_memory()
        if DEVICE.type == 'cuda':
            ResourceMonitor.check_gpu()

@app.route('/health', methods=['GET'])
@performance_logger
def health_check():
    """Health check endpoint with detailed system information"""
    model_manager = ModelManager()
    processor, model, tokenizer = model_manager.get_model_components()
    
    gpu_info = None
    if DEVICE.type == 'cuda':
        try:
            gpu_properties = torch.cuda.get_device_properties(0)
            gpu_info = {
                'name': gpu_properties.name,
                'total_memory': f"{gpu_properties.total_memory/1024**2:.2f}MB",
                'memory_allocated': f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
                'memory_reserved': f"{torch.cuda.memory_reserved()/1024**2:.2f}MB"
            }
        except Exception as e:
            logger.error(f"Error getting GPU info: {str(e)}")
    
    system_info = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model': {
            'name': MODEL_NAME,
            'loaded': None not in (processor, model, tokenizer),
            'device': DEVICE.type,
            'last_used': model_manager.last_used
        },
        'system': system_info,
        'gpu': gpu_info,
        'languages_supported': LANGUAGE_CODES
    })

def cleanup_temp_files():
    """Clean up any temporary files in the temp directory"""
    try:
        temp_dir = tempfile.gettempdir()
        pattern = Path(temp_dir) / "*.wav"
        for file in Path(temp_dir).glob("*.wav"):
            cleanup_file(str(file))
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {str(e)}")

def shutdown_handler():
    """Clean up resources during shutdown"""
    logger.info("Application shutting down...")
    try:
        model_manager = ModelManager()
        model_manager.cleanup()
        cleanup_temp_files()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {str(e)}")

# Register shutdown handler
atexit.register(shutdown_handler)

# Handle SIGTERM gracefully
def sigterm_handler(signum, frame):
    logger.info("Received SIGTERM signal")
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

if __name__ == '__main__':
    try:
        if None in ModelManager().get_model_components():
            logger.error("Could not start the app due to model loading failure")
            sys.exit(1)
        
        # Initial cleanup
        cleanup_temp_files()
        
        # Log startup information
        logger.info(f"Starting Flask app on port 5001 using {DEVICE}")
        logger.info(f"Supported languages: {list(LANGUAGE_CODES.keys())}")
        if DEVICE.type == 'cuda':
            ResourceMonitor.check_gpu()
        
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