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
import uuid
from werkzeug.utils import secure_filename

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
from services.audio_link_routes import handle_audio_url_processing
from services.audio_processor import AudioProcessor
from services.model_manager import ModelManager
from services.resource_monitor import ResourceMonitor
from services.error_handler import ErrorHandler
from services.utils import (
    cleanup_file,
    require_model,
    performance_logger
)
from services.video_routes import handle_video_processing
from services.podcast_routes import handle_podcast_upload
from services.health_routes import handle_model_health 


# Initialize environment
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
MAX_AUDIO_LENGTH = 300  # seconds (5 minutes) - for translations
MAX_PODCAST_LENGTH = 3600  # seconds (1 hour) - for podcast uploads
MAX_DURATION = 300  # seconds
MAX_TOKENS = 1500
SAMPLE_RATE = 16000
CHUNK_SIZE = 16384  # Increased for better processing
MAX_RETRIES = 3
TIMEOUT = 600  # seconds (increased for longer audio)
MEMORY_THRESHOLD = 0.9
BATCH_SIZE = 1
START_TIME = time.time()

# Upload configuration
UPLOAD_FOLDER = Path('uploads/podcasts')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}

# Create upload folder if it doesn't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    # Most common languages first
    'spa': ('spa', 'Spanish'),
    'fra': ('fra', 'French'),
    'deu': ('deu', 'German'),
    'ita': ('ita', 'Italian'),
    'por': ('por', 'Portuguese'),
    'rus': ('rus', 'Russian'),
    'jpn': ('jpn', 'Japanese'),
    'cmn': ('cmn', 'Chinese (Simplified)'),
    'cmn_Hant': ('cmn_Hant', 'Chinese (Traditional)'),
    
    # Rest in alphabetical order
    'ara': ('ara', 'Arabic'),
    'bel': ('bel', 'Belarusian'),
    'bul': ('bul', 'Bulgarian'),
    'ces': ('ces', 'Czech'),
    'dan': ('dan', 'Danish'),
    'ell': ('ell', 'Greek'),
    'epo': ('epo', 'Esperanto'),
    'est': ('est', 'Estonian'),
    'fin': ('fin', 'Finnish'),
    'heb': ('heb', 'Hebrew'),
    'hrv': ('hrv', 'Croatian'),
    'hun': ('hun', 'Hungarian'),
    'lit': ('lit', 'Lithuanian'),
    'lav': ('lav', 'Latvian'),
    'mkd': ('mkd', 'Macedonian'),
    'nld': ('nld', 'Dutch'),
    'pol': ('pol', 'Polish'),
    'ron': ('ron', 'Romanian'),
    'slk': ('slk', 'Slovak'),
    'slv': ('slv', 'Slovene'),
    'srp': ('srp', 'Serbian'),
    'tha': ('tha', 'Thai'),
    'tur': ('tur', 'Turkish'),
    'vie': ('vie', 'Vietnamese')
}

LANGUAGE_MAP = {
    'es': 'spa',
    'fr': 'fra',
    'de': 'deu',
    'it': 'ita',
    'pt': 'por',
    'ru': 'rus',
    'ja': 'jpn',
    'zh': 'cmn',
    'zh-hant': 'cmn_Hant',
    'ar': 'ara',
    'be': 'bel',
    'bg': 'bul',
    'cs': 'ces',
    'da': 'dan',
    'el': 'ell',
    'eo': 'epo',
    'et': 'est',
    'fi': 'fin',
    'he': 'heb',
    'hr': 'hrv',
    'hu': 'hun',
    'lt': 'lit',
    'lv': 'lav',
    'mk': 'mkd',
    'nl': 'nld',
    'pl': 'pol',
    'ro': 'ron',
    'sk': 'slk',
    'sl': 'slv',
    'sr': 'srp',
    'th': 'tha',
    'tr': 'tur',
    'vi': 'vie'
}

from services.translation_routes import handle_translation

# And update your process_audio_url route to handle CORS:
@app.route('/process-audio-url', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
@performance_logger
def process_audio_url():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        origin = request.headers.get('Origin', 'http://localhost:3000')
        
        response.headers.update({
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Accept, Authorization',
            'Access-Control-Allow-Credentials': 'true'
        })
        return response
    
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
            
        result = handle_audio_url_processing(url)
        
        if isinstance(result, tuple):  # Error case
            return result
            
        # Create response with audio data
        response = make_response(result['audio_data'])
        response.headers.update({
            'Content-Type': result['mime_type'],
            'Access-Control-Allow-Origin': request.headers.get('Origin', 'http://localhost:3000'),
            'Access-Control-Allow-Credentials': 'true'
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing audio URL: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add the new video processing route here
@app.route('/process-video', methods=['POST', 'OPTIONS'])
@limiter.limit("2 per minute")
@performance_logger
def process_video():
    try:
        target_language = request.form.get('target_language', 'fra')
        return handle_video_processing(target_language)
    except Exception as e:
        return ErrorHandler.handle_error(e)



@app.route('/translate', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
@require_model
@performance_logger
def translate_audio_endpoint():
    global processor, model, text_model, tokenizer
    TRANSLATION_REQUESTS.inc()
    
    try:
        return handle_translation(
            processor, model, text_model, tokenizer,
            DEVICE, SAMPLE_RATE, LANGUAGE_MAP, LANGUAGE_CODES
        )
    except Exception as e:
        TRANSLATION_ERRORS.inc()
        return ErrorHandler.handle_error(e)

@app.route('/upload_podcast', methods=['POST', 'OPTIONS'])
@limiter.limit("5 per minute")
@performance_logger
def upload_podcast():
    return handle_podcast_upload(UPLOAD_FOLDER, MAX_PODCAST_LENGTH, ALLOWED_EXTENSIONS)
            
@app.route('/health/model', methods=['GET'])
@performance_logger
def model_health():
    return handle_model_health(DEVICE)


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


def cleanup_old_podcasts():
    """Clean up podcast files older than 24 hours"""
    try:
        current_time = time.time()
        for file_path in UPLOAD_FOLDER.glob('*'):
            if file_path.stat().st_mtime < (current_time - 86400):  # 24 hours
                cleanup_file(file_path)
                logger.info(f"Removed old podcast file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up old podcasts: {str(e)}")

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
        cleanup_old_podcasts()  # Added this line
        
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