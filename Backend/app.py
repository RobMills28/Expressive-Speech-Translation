# app.py

import os
import sys
import time
import atexit
import signal
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
import gc
import warnings

import subprocess
from werkzeug.utils import secure_filename


# --- Warnings and Environment Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning, module="espnet2") # If you still have ESPnet code
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*transformers.deepspeed module is deprecated.*", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false" # For Hugging Face tokenizers

# --- Logging Configuration ---
LOG_LEVEL_CONSOLE = logging.INFO
LOG_LEVEL_FILE_MAIN = logging.INFO
LOG_LEVEL_FILE_DEBUG = logging.DEBUG
LOG_LEVEL_FILE_ERROR = logging.ERROR

DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d (%(funcName)s)] - %(message)s'
)

def extract_audio_from_any_file(media_file_path: Path, output_wav_path: Path) -> Path:
    """A robust helper function to extract audio from any media file using ffmpeg."""
    command = [
        'ffmpeg', '-y', '-i', str(media_file_path),
        '-vn',  # This flag strips any video, leaving only the audio.
        '-acodec', 'pcm_s16le',  # Standard, uncompressed WAV format.
        '-ar', '16000',          # The sample rate your ML models expect.
        '-ac', '1',              # Mono audio.
        str(output_wav_path)
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        if not output_wav_path.exists() or output_wav_path.stat().st_size == 0:
            raise IOError("FFmpeg ran but created an empty or missing output file.")
        logger.info(f"Successfully extracted audio to {output_wav_path}")
        return output_wav_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error during audio extraction: {e.stderr}")
        raise IOError(f"Failed to extract audio. FFmpeg returned an error.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio extraction: {e}")
        raise

def initial_logging_setup():
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        print(f"Initial Logging: Root logger has {len(root_logger.handlers)} handlers. Removing ALL for a clean slate.")
        for handler in list(root_logger.handlers):
            try: handler.close(); root_logger.removeHandler(handler)
            except Exception as e: print(f"Initial Logging: Error removing/closing handler {handler}: {e}")
    else:
        print("Initial Logging: Root logger has no pre-existing handlers.")

    root_logger.setLevel(LOG_LEVEL_FILE_DEBUG) # Set root logger to lowest level

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(DETAILED_FORMATTER)
    console_handler.setLevel(LOG_LEVEL_CONSOLE)
    root_logger.addHandler(console_handler)
    print(f"Initial Logging: Console handler added with level {logging.getLevelName(LOG_LEVEL_CONSOLE)}.")

    log_dir = Path('logs'); log_dir.mkdir(exist_ok=True)
    debug_log_path = log_dir / 'app_debug.log'
    debug_file_handler = TimedRotatingFileHandler(debug_log_path, when='midnight', interval=1, backupCount=3, encoding='utf-8')
    debug_file_handler.setFormatter(DETAILED_FORMATTER); debug_file_handler.setLevel(LOG_LEVEL_FILE_DEBUG)
    root_logger.addHandler(debug_file_handler)
    print(f"Initial Logging: Debug file handler added ({debug_log_path.name}) with level {logging.getLevelName(LOG_LEVEL_FILE_DEBUG)}.")

    main_log_path = log_dir / 'app_main.log'
    main_file_handler = TimedRotatingFileHandler(main_log_path, when='midnight', interval=1, backupCount=7, encoding='utf-8')
    main_file_handler.setFormatter(DETAILED_FORMATTER); main_file_handler.setLevel(LOG_LEVEL_FILE_MAIN)
    root_logger.addHandler(main_file_handler)
    print(f"Initial Logging: Main file handler added ({main_log_path.name}) with level {logging.getLevelName(LOG_LEVEL_FILE_MAIN)}.")

    error_log_path = log_dir / 'app_error.log'
    error_file_handler = RotatingFileHandler(error_log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    error_file_handler.setFormatter(DETAILED_FORMATTER); error_file_handler.setLevel(LOG_LEVEL_FILE_ERROR)
    root_logger.addHandler(error_file_handler)
    print(f"Initial Logging: Error file handler added ({error_log_path.name}) with level {logging.getLevelName(LOG_LEVEL_FILE_ERROR)}.")

    libraries_config = {
        'boto3': logging.WARNING, # Quieter by default unless debugging Polly
        'botocore': logging.WARNING,
        's3transfer': logging.WARNING,
        'werkzeug': logging.WARNING,
        'pydub': logging.WARNING,
        'matplotlib': logging.WARNING,
        'urllib3': logging.INFO,
        'asyncio': logging.INFO,
        'h5py': logging.WARNING,
        'numba': logging.WARNING, # Whisper uses numba
        'whisper': logging.INFO,
        'transformers.tokenization_utils_base': logging.WARNING,
        'transformers': logging.INFO,
        'torchaudio': logging.INFO,
        'TTS': logging.INFO, # For Coqui TTS (if it were used locally)
        'httpx': logging.INFO, # For Gradio client if used
        'multipart': logging.INFO,
        'speechbrain': logging.INFO, # For similarity checker
    }
    for lib_name, level in libraries_config.items():
        logging.getLogger(lib_name).setLevel(level)
        logging.debug(f"Initial Logging: Set log level for '{lib_name}' to {logging.getLevelName(level)}.")

    logging.getLogger('services').setLevel(logging.DEBUG) # Your services modules
    logging.debug(f"Initial Logging: Set log level for parent 'services' logger to DEBUG.")

    logging.info("Global logging setup complete from app.py. Root effective level: %s. Console set to: %s.",
                 logging.getLevelName(logging.getLogger().getEffectiveLevel()),
                 logging.getLevelName(LOG_LEVEL_CONSOLE))

initial_logging_setup()

logger = logging.getLogger(__name__) # This is the __main__ logger
logger.info("--- app.py module logger initialized ---")

# --- Core Imports ---
import hashlib
import tempfile
import json
import base64
import traceback
from datetime import datetime
import uuid # For request IDs if needed elsewhere
from werkzeug.utils import secure_filename
import psutil # For health checks
import torch
import torchaudio # For audio saving

# --- Flask and Extensions ---
from flask import Flask, request, jsonify, make_response, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

logger.info("--- app.py: Starting service module imports ---")
from services.audio_link_routes import handle_audio_url_processing # Assuming this is updated or not CosyVoice dependent
from services.audio_processor import AudioProcessor # General audio utilities
APP_WIDE_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Application-wide ML device set to: {APP_WIDE_DEVICE}")

from services.error_handler import ErrorHandler
from services.utils import cleanup_file, performance_logger
from services.video_routes import handle_video_processing
from services.podcast_routes import handle_podcast_upload # Assuming this is updated or not CosyVoice dependent
from services.health_routes import handle_model_health
from services.translation_strategy import TranslationManager
from services.cascaded_backend import CascadedBackend # This will be the CosyVoice API calling version
logger.info("--- app.py: Core service module imports complete ---")

# Optional ESPnet backend (if you still want to support it as an alternative)
try:
    from services.espnet_backend import ESPnetBackend
    ESPNET_AVAILABLE = True; logger.info("ESPnet backend components potentially available.")
except ImportError:
    logger.warning("ESPnet native library not found or import failed. ESPnetBackend will NOT be available.")
    ESPNET_AVAILABLE = False
except Exception as e_espnet:
    logger.error(f"Unexpected error during ESPnetBackend import: {e_espnet}", exc_info=True)
    ESPNET_AVAILABLE = False

load_dotenv(); logger.info(".env file loaded if present.")

# --- Application Constants ---
MAX_AUDIO_LENGTH_SECONDS = 300  # Max length for direct audio translation (5 minutes)
MAX_PODCAST_LENGTH_SECONDS = 3600 # Max length for podcast (1 hour)
MAX_VIDEO_MB = 150 # Max video upload size in MB
SAMPLE_RATE = 16000 # Standard sample rate for ASR and final audio output tensor

UPLOAD_FOLDER_PODCASTS = Path('uploads/podcasts')
ALLOWED_PODCAST_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac'}
UPLOAD_FOLDER_PODCASTS.mkdir(parents=True, exist_ok=True)


# --- Application Setup ---
logger.info("Creating TranslationManager instance..."); translation_manager = TranslationManager()

# HuggingFace Token (mainly for NLLB or other HF model downloads if not cached)
hf_auth_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_auth_token:
    logger.warning("HUGGINGFACE_TOKEN environment variable not set. Some model downloads may fail.")

logger.info("Attempting to instantiate and register default backend (CascadedBackend for CosyVoice API)...")
cascaded_backend_instance = None # Define for finally block
try:
    cascaded_backend_instance = CascadedBackend(device=APP_WIDE_DEVICE) # use_voice_cloning is implicit now
    translation_manager.register_backend("cascaded", cascaded_backend_instance, is_default=True)
except Exception as e_backend_reg:
    logger.critical(f"CRITICAL FAILURE: Could not instantiate/register CascadedBackend: {e_backend_reg}", exc_info=True)
    sys.exit(1) # Exit if backend cannot be registered

app = Flask(__name__); logger.info("Flask app instance created.")
CORS(app, resources={ r"/*": { "origins": ["http://localhost:3000", "http://localhost:3001"], "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Accept", "Authorization", "X-Requested-With", "Range", "Accept-Ranges", "Origin"], "expose_headers": ["Content-Type", "Content-Length", "Content-Range", "Content-Disposition", "Accept-Ranges", "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"], "supports_credentials": True, "max_age": 120, "automatic_options": True }})
logger.info("CORS configured.")
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["500 per day", "100 per hour"]) # Generous limits for testing
logger.info("Flask-Limiter initialized.")
app.config['IS_SHUTTING_DOWN'] = False
app.config['MAX_CONTENT_LENGTH'] = MAX_VIDEO_MB * 1024 * 1024 # Set max content length for uploads

# --- Request Handling Middleware ---
@app.before_request
def before_request_middleware():
    if request.method == 'OPTIONS': return # Allow CORS preflight
    # Basic validation (can be expanded)
    if request.method not in ['GET', 'POST', 'HEAD']:
        logger.warning(f"Method not allowed: {request.method} for {request.path}")
        return jsonify({'error': 'Method not allowed'}), 405
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type', '').lower()
        # Allow JSON and multipart/form-data for POST requests
        if not content_type.startswith('multipart/form-data') and \
           not content_type.startswith('application/json'):
            # Check if it's an empty POST or truly invalid content type
            if not (request.form or request.files or request.is_json):
                 logger.warning(f"Invalid content type for POST: '{content_type}' for {request.path}.")
                 return jsonify({'error': 'Invalid content type for POST. Expected JSON or multipart/form-data.'}), 400
    request.start_time = time.time() # For performance logging

@app.after_request
def after_request_middleware(response: Response) -> Response:
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"REQ: {request.method} {request.path} ({request.remote_addr}) -> RSP: {response.status_code} [{duration:.2f}s]")
    else: # Should not happen if before_request runs
        logger.info(f"REQ: {request.method} {request.path} ({request.remote_addr}) -> RSP: {response.status_code} [duration N/A]")
    return response

@app.errorhandler(Exception)
def central_error_handler(e: Exception):
    # Log the full traceback for any unhandled exception
    logger.error(f"Unhandled Exception: {request.method} {request.path}", exc_info=True)
    # Use your custom ErrorHandler
    error_response, status_code = ErrorHandler.handle_error(e); return error_response, status_code


# --- API Routes ---
@app.route('/translate', methods=['POST', 'OPTIONS'])
@limiter.limit("20 per minute")
@performance_logger
def translate_audio_route():
    if request.method == 'OPTIONS':
        return make_response() # Handle CORS preflight
    
    # Use a single, self-cleaning temporary directory for all files for this request.
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_dir_path = Path(temp_dir)
            
            # 1. Get the correct backend for the translation task.
            backend_to_use = translation_manager.get_backend() 
            logger.info(f"Using backend: {type(backend_to_use).__name__} for /translate")
            
            # 2. Validate the incoming request for file and language.
            if 'file' not in request.files:
                return ErrorHandler.format_validation_error('No file part found in request.')
            file = request.files['file']
            if not file or not file.filename:
                return ErrorHandler.format_validation_error('No file selected or filename is empty.')
            
            target_language_app_code = request.form.get('target_language', 'fra') # Default to French
            logger.info(f"/translate request: Target='{target_language_app_code}' for file '{file.filename}'")

            # 3. Save the uploaded file (regardless of type) to a temporary location.
            original_filename = secure_filename(file.filename)
            original_temp_path = temp_dir_path / original_filename
            file.save(str(original_temp_path))

            # 4. CRITICAL STEP: Always extract audio to a clean, standardized WAV file.
            # This makes the pipeline robust to any input file type (mp4, mov, mp3, etc.).
            processed_wav_path = temp_dir_path / "processed_for_pipeline.wav"
            extract_audio_from_any_file(original_temp_path, processed_wav_path)

            # 5. Perform all validations and processing on the GUARANTEED clean WAV file.
            audio_processor = AudioProcessor()
            is_valid_length, length_error_msg = audio_processor.validate_audio_length(str(processed_wav_path), max_length_seconds=MAX_AUDIO_LENGTH_SECONDS)
            if not is_valid_length:
                 logger.error(f"Audio length validation failed for {processed_wav_path}: {length_error_msg}")
                 return ErrorHandler.format_validation_error(length_error_msg)
            
            # Process the clean audio file into a tensor for the model.
            audio_tensor = audio_processor.process_audio(str(processed_wav_path))
            if not audio_processor.is_valid_audio(audio_tensor):
                logger.error(f"Processed audio tensor is invalid (e.g., silent) for {processed_wav_path}")
                return ErrorHandler.handle_error(ValueError("Processed audio is invalid or silent."))

            # 6. Perform the translation using the selected backend.
            result = backend_to_use.translate_speech(
                audio_tensor=audio_tensor,
                source_lang="eng",
                target_lang=target_language_app_code
            )
            
            translated_audio_tensor = result["audio"] 
            
            # 7. Prepare the final audio file and encode it for the JSON response.
            final_audio_path = temp_dir_path / "final_output.wav"
            
            # Ensure the tensor has the correct shape for torchaudio.save
            if translated_audio_tensor.ndim == 1:
                translated_audio_tensor = translated_audio_tensor.unsqueeze(0)
            if translated_audio_tensor.shape[0] > 1 and translated_audio_tensor.shape[1] == 1:
                translated_audio_tensor = translated_audio_tensor.transpose(0, 1)

            torchaudio.save(str(final_audio_path), translated_audio_tensor.cpu(), SAMPLE_RATE)
            
            with open(final_audio_path, 'rb') as audio_data_obj:
                audio_b64 = base64.b64encode(audio_data_obj.read()).decode('utf-8')
        
            # 8. Return the final JSON payload to the frontend.
            return jsonify({
                'audio': audio_b64, 
                'transcripts': result["transcripts"]
            })

        except Exception as e:
            logger.error(f"Error in /translate route: {e}", exc_info=True)
            return ErrorHandler.handle_error(e)

@app.route('/process-video', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute") # Adjusted for video
@performance_logger
def process_video_route():
    # ... (video route logic, largely unchanged but backend instance will be the CosyVoice API one)
    logger.info(f"Request for /process-video (Method: {request.method})")
    if request.method == 'OPTIONS': logger.info("Handling OPTIONS for /process-video"); resp = make_response(); return resp
    try:
        logger.info(f"Content-Type for /process-video POST: {request.content_type}")
        if 'video' not in request.files:
            logger.error("No 'video' file part in request.files for /process-video")
            def err_stream(): yield f"data: {json.dumps({'error': 'No video file part in request.', 'phase': 'File Upload Error'})}\n\n"
            return Response(stream_with_context(err_stream()),mimetype='text/event-stream', status=400)
        video_file = request.files['video']
        if not video_file or not video_file.filename:
            logger.error("No video file selected or filename empty for /process-video")
            def err_stream_fn(): yield f"data: {json.dumps({'error': 'No video file selected or filename empty.', 'phase': 'File Upload Error'})}\n\n"
            return Response(stream_with_context(err_stream_fn()),mimetype='text/event-stream', status=400)
        
        target_language = request.form.get('target_language', 'fra') 
        apply_lip_sync_str = request.form.get('apply_lip_sync', 'true') 
        apply_lip_sync_bool = apply_lip_sync_str.lower() == 'true'
        
        form_voice_cloning_str = request.form.get('use_voice_cloning', 'true') # Frontend still sends this
        logger.info(f"Video processing: Target Lang='{target_language}', Apply Lip Sync='{apply_lip_sync_bool}', Form 'use_voice_cloning'='{form_voice_cloning_str}' (Note: CosyVoice API backend handles cloning implicitly based on reference audio sent)")

        video_processing_backend = translation_manager.get_backend('cascaded') 
        if not video_processing_backend.initialized:
            logger.error("Failed to get/initialize CascadedBackend (for CosyVoice API) for video in /process-video.")
            def err_stream_backend(): yield f"data: {json.dumps({'error': 'Backend service not ready.', 'phase': 'Backend Error'})}\n\n"
            return Response(stream_with_context(err_stream_backend()), mimetype='text/event-stream', status=503)
        
        # The use_voice_cloning_config on CascadedBackend might not be directly used if it's always cloning via API
        # but it's good that the frontend can signal intent.
        if hasattr(video_processing_backend, 'use_voice_cloning_config'):
            video_processing_backend.use_voice_cloning_config = form_voice_cloning_str == 'true'


        return handle_video_processing(request.files, target_language, video_processing_backend, apply_lip_sync_bool)
    except Exception as e:
        logger.error(f"Error in /process-video route: {e}", exc_info=True)
        def err_stream_exc():
            err_payload = json.dumps({'error': 'Internal server error during video processing.','phase': 'Server Error','details': str(e)[:200]})
            yield f"data: {err_payload}\n\n"
        return Response(stream_with_context(err_stream_exc()), mimetype='text/event-stream', status=500)


@app.route('/available-backends', methods=['GET'])
def available_backends_route():
    # ... (no change)
    try:
        backends_dict = translation_manager.get_available_backends(); backend_names = list(backends_dict.keys())
        default_backend_name = translation_manager.default_backend
        return jsonify({'backends': backend_names, 'default': default_backend_name})
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/supported-languages', methods=['GET'])
def supported_languages_route():
    # ... (no change)
    try:
        backend_name_query = request.args.get('backend') 
        backend_to_check = translation_manager.get_backend(backend_name_query) 
        return jsonify({'languages': backend_to_check.get_supported_languages()})
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/process-audio-url', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
@performance_logger
def process_audio_url_route():
    # ... (Pass translation_manager to handle_audio_url_processing)
    if request.method == 'OPTIONS': resp = make_response(); return resp
    try:
        data = request.get_json();
        if not data or 'url' not in data: return ErrorHandler.format_validation_error('No URL provided')
        url = data['url']
        result = handle_audio_url_processing(url, translation_manager) 
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int): return jsonify(result[0]), result[1]
        if 'error' in result and isinstance(result, dict) : return jsonify(result), result.get('status_code', 400)
        response = make_response(result['audio_data']); response.headers['Content-Type'] = result['mime_type']
        return response
    except Exception as e: return ErrorHandler.handle_error(e)


@app.route('/translation-service-status', methods=['GET'])
def translation_service_status_route():
    # ... (no change from previous XTTS version)
    try:
        default_backend = translation_manager.get_backend() 
        is_ready = default_backend.initialized
        backend_type = type(default_backend).__name__
        
        status_message = f"{backend_type} is initialized."
        # Specific check for CosyVoice API readiness could be added if _check_cosyvoice_api_status exposed a public status
        # For now, `initialized` implies the API was healthy during its init.
        if hasattr(default_backend, '_check_cosyvoice_api_status') and not default_backend._check_cosyvoice_api_status():
             is_ready = False
             status_message = f"{backend_type} initialized, but downstream CosyVoice API is not healthy."

        return jsonify({
            'service_status': 'ready' if is_ready else 'not_ready',
            'backend_type': backend_type,
            'message': status_message
        })
    except Exception as e:
        logger.error(f"Error checking translation service status: {e}", exc_info=True)
        return jsonify({'service_status': 'error', 'message': f"Error: {str(e)}"}), 500


@app.route('/upload_podcast', methods=['POST', 'OPTIONS'])
@limiter.limit("5 per minute") 
@performance_logger
def upload_podcast_route():
    # ... (Pass translation_manager to handle_podcast_upload)
    if request.method == 'OPTIONS': resp = make_response(); return resp
    return handle_podcast_upload(UPLOAD_FOLDER_PODCASTS, MAX_PODCAST_LENGTH_SECONDS, ALLOWED_PODCAST_EXTENSIONS, translation_manager)

@app.route('/health/model', methods=['GET'])
def model_health_route():
    # ... (no change)
    return handle_model_health(APP_WIDE_DEVICE, translation_manager, None) 


# --- Graceful Shutdown Logic ---
def _app_cleanup_tasks():
    logger.info("Shutdown: Cleaning up application resources...")
    try:
        logger.info("ModelManager cleanup skipped as it has been removed from the application.")

        if 'translation_manager' in globals() and translation_manager.backends:
            logger.info("Cleaning up translation backends...")
            for name, backend in translation_manager.get_available_backends().items():
                if hasattr(backend, 'cleanup') and callable(backend.cleanup):
                    logger.info(f"Calling .cleanup() for backend: {name}");
                    try: backend.cleanup()
                    except Exception as e_clean: logger.error(f"Error cleaning backend {name}: {e_clean}", exc_info=True)
        else: logger.info("TranslationManager or its backends not found, skipping backend cleanup.")
        
        # Check if APP_WIDE_DEVICE is CUDA before trying to empty cache
        if isinstance(APP_WIDE_DEVICE, torch.device) and APP_WIDE_DEVICE.type == 'cuda' and torch.cuda.is_available(): 
            logger.info("Emptying CUDA cache..."); torch.cuda.empty_cache()
        
        gc.collect(); logger.info("Resource cleanup by _app_cleanup_tasks complete.")
    except Exception as e: logger.error(f"Error during _app_cleanup_tasks: {e}", exc_info=True)

def _graceful_shutdown_signal_handler(_signum, _frame):
    logger.info(f"Signal {_signum} received, initiating graceful shutdown...")
    if not app.config.get('IS_SHUTTING_DOWN', False):
        app.config['IS_SHUTTING_DOWN'] = True
        logger.info("Graceful shutdown tasks to be run by atexit handler. Exiting now to trigger atexit.")
        # Note: sys.exit(0) in a signal handler might not always trigger atexit in all threaded scenarios.
        # A more robust way might involve setting a flag and having the main Flask loop check it,
        # or using a different shutdown mechanism if issues arise. For now, this is standard.
        sys.exit(0) 

atexit.register(_app_cleanup_tasks)
signal.signal(signal.SIGTERM, _graceful_shutdown_signal_handler)
signal.signal(signal.SIGINT, _graceful_shutdown_signal_handler)
# --- End Graceful Shutdown Logic ---

if __name__ == '__main__':
    try:
        logger.info("="*50);
        logger.info("Starting Magenta AI Backend Main Application (app.py - Targeting CosyVoice API)")
        logger.info(f"Running with Python version: {sys.version.splitlines()[0]}")
        if not os.getenv('HUGGINGFACE_TOKEN'): 
            logger.warning("HUGGINGFACE_TOKEN environment variable not set. Some model downloads may fail.")
        
        logger.info(f"Attempting to initialize default S2ST backend: '{translation_manager.default_backend or 'Cascaded (default will be set)'}'")
        
        # This block will trigger initialization of CascadedBackend, which includes CosyVoice API health check.
        default_backend_instance = translation_manager.get_backend() 
        
        if not default_backend_instance.initialized:
             logger.critical(
                 f"CRITICAL FAILURE: Default backend '{translation_manager.default_backend}' "
                 f"(type: {type(default_backend_instance).__name__}) "
                 f"DID NOT successfully initialize. Check logs from its initialization routine, "
                 f"especially regarding dependent API health (e.g., CosyVoice API at {os.getenv('COSYVOICE_API_URL')})."
             )
             sys.exit(1) # Exit if backend isn't initialized (e.g. CosyVoice API unhealthy)
        
        logger.info(
            f"Default S2ST backend '{translation_manager.default_backend}' "
            f"(type: {type(default_backend_instance).__name__}) "
            f"confirmed initialized (dependent APIs were healthy during init)."
        )
        
        logger.info(f"Flask app starting on port 5001. ML Processing Device: {APP_WIDE_DEVICE}")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)
    except SystemExit: 
        logger.info("Application exiting due to SystemExit (likely signal handler or startup check failure).")
    except Exception as e_startup:
        logger.critical(f"FATAL: Application startup sequence failed: {e_startup}", exc_info=True)
    finally:
        logger.info("Application shutdown sequence initiated or completed.")