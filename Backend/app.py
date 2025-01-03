# Standard library imports
import os
import sys
import time
import atexit
import signal
import hashlib
import warnings
import tempfile
import json
import base64
import traceback  # Add this for the improved error handling
from datetime import datetime
from pathlib import Path

# Third-party imports: Audio processing
from pydub import AudioSegment  # For MP3 conversion

# Third-party imports: Monitoring and Metrics
from prometheus_client import Counter, Histogram, start_http_server
import psutil

# Third-party imports: ML and Audio
import torch
import numpy as np
import torchaudio

# Flask and Extensions
from flask import (
    Flask, 
    request, 
    jsonify,
    make_response, 
    Response
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

# Local imports
from services.audio_processor import AudioProcessor
from services.model_manager import ModelManager
from services.resource_monitor import ResourceMonitor
from services.error_handler import ErrorHandler
from services.utils import (
    cleanup_file,
    require_model,
    performance_logger
)

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

# Global model components - Added this section
global processor, model, text_model, tokenizer
processor = None
model = None
text_model = None
tokenizer = None

# Initialize Flask app
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

@app.route('/translate', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
@require_model
@performance_logger
def translate_audio_endpoint():
    global processor, model, text_model, tokenizer
    TRANSLATION_REQUESTS.inc()
    
    if request.method == 'OPTIONS':
        return '', 204
    
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    start_time = time.time()
    temp_files = []
    source_text = "Text extraction unavailable"
    target_text = "Text extraction unavailable"
    
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

        # Initialize audio processor with validation
        try:
            audio_processor = AudioProcessor()
            if not audio_processor.diagnostics:
                logger.warning(f"Request {request_id}: AudioDiagnostics not initialized properly")
        except Exception as e:
            logger.error(f"Failed to initialize AudioProcessor: {str(e)}")
            return jsonify({'error': 'Internal processing error'}), 500

        # Save and process audio file with proper error handling
        file_extension = Path(file.filename).suffix.lower()
        
        try:
            if file_extension == '.mp3':
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                    temp_files.append(wav_file.name)
                    file.save(wav_file.name + '.mp3')
                    temp_files.append(wav_file.name + '.mp3')
                    sound = AudioSegment.from_mp3(wav_file.name + '.mp3')
                    sound.export(wav_file.name, format='wav')
                    audio_path = wav_file.name
            else:
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_input:
                    temp_files.append(temp_input.name)
                    file.save(temp_input.name)
                    audio_path = temp_input.name

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise ValueError("Failed to save audio file")
            
            logger.info(f"Request {request_id}: Saved input file: {audio_path}")
        except Exception as e:
            logger.error(f"File handling error: {str(e)}")
            return jsonify({'error': 'Failed to process uploaded file'}), 400

        is_valid, error_message = audio_processor.validate_audio_length(audio_path)
        if not is_valid:
            logger.error(f"Request {request_id}: Audio validation failed - {error_message}")
            return jsonify({'error': error_message}), 400

        try:
            logger.info(f"Request {request_id}: Beginning audio processing with diagnostics")
            
            audio, input_diagnostics = audio_processor.process_audio_enhanced(
                audio_path,
                target_language=model_language,
                return_diagnostics=True
            )
            
            if input_diagnostics:
                input_report = audio_processor.diagnostics.generate_report(input_diagnostics, model_language)
                if input_report:
                    logger.info("\n" + "="*50)
                    logger.info("Input Audio Quality Report")
                    logger.info("="*50)
                    logger.info(f"Request ID: {request_id}")
                    logger.info(f"Target Language: {model_language}")
                    logger.info("-"*50)
                    logger.info(input_report)
                    logger.info("="*50 + "\n")
                
                if 'metrics' in input_diagnostics:
                    for metric, value in input_diagnostics['metrics'].items():
                        logger.debug(f"Input Quality Metric - {metric}: {value}")
            
            audio_numpy = audio.squeeze().numpy()
            
            if np.isnan(audio_numpy).any() or np.isinf(audio_numpy).any():
                raise ValueError("Invalid audio data detected")
                
            logger.info(f"Request {request_id}: Audio processed successfully")
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return jsonify({'error': f'Failed to process audio: {str(e)}'}), 400
        
        try:
            # IMPROVED: Enhanced input processing for better transcription
            inputs = processor(
                audios=audio_numpy,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                src_lang="eng",
                tgt_lang=model_language,
                padding=True,              # Add padding
                truncation=True,           # Enable truncation if needed
                max_length=256000         # Set reasonable max length for audio
            )
            
            logger.info(f"Input keys: {inputs.keys()}")
            
            if not inputs or not inputs.keys():
                raise ValueError("Failed to prepare model inputs")
            
            if DEVICE.type == 'cuda':
                inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

            # IMPROVED: Enhanced source text generation (English transcription)
            try:
                with torch.no_grad():
                    # First pass - get initial transcription
                    source_outputs = text_model.generate(
                        input_features=inputs["input_features"],
                        tgt_lang="eng",
                        num_beams=8,                    # Increased beam search
                        do_sample=False,                # Disable sampling for accurate transcription
                        max_new_tokens=1000,            # Longer output allowed
                        temperature=0.2,                # Lower temperature for more precise recognition
                        length_penalty=1.0,             # Neutral length penalty
                        repetition_penalty=1.5,         # Keep repetition penalty
                        no_repeat_ngram_size=3          # Keep n-gram blocking
                    )
                    source_text = processor.batch_decode(source_outputs, skip_special_tokens=True)[0]
                    
                    # Check for potential transcription issues
                    if any(phrase in source_text for phrase in [" H. H. H", ", the, the", " of the, of the"]):
                        logger.info("Detected potential repetition, performing second pass")
                        source_outputs = text_model.generate(
                            input_features=inputs["input_features"],
                            tgt_lang="eng",
                            num_beams=8,                    
                            do_sample=False,               
                            max_new_tokens=1000,            
                            repetition_penalty=2.0,        # Increased repetition penalty for second pass
                            no_repeat_ngram_size=4,        # Increased n-gram blocking
                            temperature=0.2                # Keep low temperature
                        )
                        source_text = processor.batch_decode(source_outputs, skip_special_tokens=True)[0]
                    
                    logger.info(f"Source text: {source_text}")
                    
            except Exception as e:
                logger.error(f"Source text generation error: {str(e)}")
                source_text = "Text extraction unavailable"

            # Generate target text using same input features for consistency
            try:
                with torch.no_grad():
                    target_outputs = text_model.generate(
                        input_features=inputs["input_features"],
                        tgt_lang=model_language,
                        num_beams=4,
                        max_new_tokens=1000,             # Match source text length
                        length_penalty=0.8,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=3
                    )
                    target_text = processor.batch_decode(target_outputs, skip_special_tokens=True)[0]
                    logger.info(f"Target text: {target_text}")
            except Exception as e:
                logger.error(f"Target text generation error: {str(e)}")
                target_text = "Text extraction unavailable"

            # Generate the audio
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    tgt_lang=model_language,
                    num_beams=5,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    length_penalty=1.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Handle outputs based on type
            if isinstance(outputs, tuple):
                logger.info("Processing tuple output from model")
                audio_output = outputs[0].cpu().numpy()
            else:
                logger.info("Processing tensor output from model")
                audio_output = outputs.cpu().numpy()

            if audio_output is None or audio_output.size == 0:
                raise ValueError("No audio data generated")

            if np.abs(audio_output).max() > 1.0:
                audio_output = audio_output / np.abs(audio_output).max()

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return jsonify({'error': f'Translation failed: {str(e)}'}), 500

        try:
            output_tensor = torch.from_numpy(audio_output).unsqueeze(0)
            output_diagnostics = audio_processor.diagnostics.analyze_translation(output_tensor, model_language)
            
            if output_diagnostics:
                output_report = audio_processor.diagnostics.generate_report(output_diagnostics, model_language)
                if output_report:
                    logger.info("\n" + "="*50)
                    logger.info("Translated Audio Quality Report")
                    logger.info("="*50)
                    logger.info(f"Request ID: {request_id}")
                    logger.info(f"Target Language: {model_language}")
                    logger.info("-"*50)
                    logger.info(output_report)
                    logger.info("="*50 + "\n")
                    
                    if 'metrics' in output_diagnostics:
                        for metric, value in output_diagnostics['metrics'].items():
                            logger.debug(f"Quality Metric - {metric}: {value}")
            
            logger.info(f"Request {request_id}: Audio output processed successfully")
            
        except Exception as e:
            logger.error(f"Diagnostics error for request {request_id}: {str(e)}\n"
                      f"Traceback: {traceback.format_exc()}")
            logger.warning("Continuing with translation despite diagnostics failure")

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_files.append(temp_output.name)
                
                # Ensure audio output is properly shaped
                waveform_tensor = torch.tensor(audio_output)
                if len(waveform_tensor.shape) == 1:
                    waveform_tensor = waveform_tensor.unsqueeze(0)
                
                torchaudio.save(
                    temp_output.name,
                    waveform_tensor,
                    sample_rate=16000
                )
                
                if not os.path.exists(temp_output.name) or os.path.getsize(temp_output.name) == 0:
                    raise ValueError("Failed to save translated audio")
                
                with open(temp_output.name, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                if not audio_data:
                    raise ValueError("Generated audio data is empty")
                
                response_data = {
                    'audio': base64.b64encode(audio_data).decode('utf-8'),
                    'transcripts': {
                        'source': source_text,
                        'target': target_text
                    }
                }

                response = Response(
                    json.dumps(response_data),
                    mimetype='application/json',
                    headers={
                        'Content-Type': 'application/json',
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
        # Cleanup
        for temp_file in temp_files:
            cleanup_file(temp_file)
        
        duration = time.time() - start_time
        logger.info(f"Request {request_id} completed in {duration:.2f}s")
        PROCESSING_TIME.observe(duration)
        
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
        processor, model, text_model, tokenizer = model_manager.get_model_components()
        if None in (processor, model, text_model, tokenizer):
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