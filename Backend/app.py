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
import warnings # Import the warnings module

# --- Suppress Specific FutureWarning categories ---
warnings.filterwarnings("ignore", category=FutureWarning, module="espnet2")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*transformers.deepspeed module is deprecated.*", category=FutureWarning)
# The "Failed to import Flash Attention" is a UserWarning or similar, not a FutureWarning.
# If you want to suppress it:
# warnings.filterwarnings("ignore", message=".*Failed to import Flash Attention.*")
# However, it's often better to address the underlying optional dependency if possible or confirm it's truly benign.

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOG_LEVEL_CONSOLE = logging.INFO
LOG_LEVEL_FILE_MAIN = logging.INFO
LOG_LEVEL_FILE_DEBUG = logging.DEBUG
LOG_LEVEL_FILE_ERROR = logging.ERROR

DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d (%(funcName)s)] - %(message)s'
)

def initial_logging_setup():
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        print(f"Initial Logging: Root logger has {len(root_logger.handlers)} handlers. Removing ALL for a clean slate.")
        for handler in list(root_logger.handlers):
            try: handler.close(); root_logger.removeHandler(handler)
            except Exception as e: print(f"Initial Logging: Error removing/closing handler {handler}: {e}")
    else:
        print("Initial Logging: Root logger has no pre-existing handlers.")

    root_logger.setLevel(LOG_LEVEL_FILE_DEBUG)

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
        'boto3': logging.DEBUG,
        'botocore': logging.DEBUG, # This includes botocore.credentials
        's3transfer': logging.INFO,
        'werkzeug': logging.WARNING,
        'pydub': logging.WARNING,
        'matplotlib': logging.WARNING,
        'urllib3': logging.INFO,
        'asyncio': logging.INFO,
        'h5py': logging.WARNING,
        'numba': logging.WARNING,
        'whisper': logging.INFO,
        'transformers.tokenization_utils_base': logging.WARNING,
        'transformers': logging.INFO,
        'torchaudio': logging.INFO,
    }
    for lib_name, level in libraries_config.items():
        logging.getLogger(lib_name).setLevel(level)
        logging.debug(f"Initial Logging: Set log level for '{lib_name}' to {logging.getLevelName(level)}.")

    # **** KEY CHANGE HERE ****
    # Explicitly set the log level for your application's 'services' package logger.
    # This ensures all loggers like 'services.cascaded_backend', 'services.audio_processor'
    # will process DEBUG messages and pass them to the root handlers.
    logging.getLogger('services').setLevel(logging.DEBUG)
    logging.debug(f"Initial Logging: Set log level for parent 'services' logger to DEBUG.")
    # You can also be more specific if needed:
    # logging.getLogger('services.cascaded_backend').setLevel(logging.DEBUG)

    logging.info("Global logging setup complete from app.py. Root effective level: %s. Console set to: %s.",
                 logging.getLevelName(logging.getLogger().getEffectiveLevel()),
                 logging.getLevelName(LOG_LEVEL_CONSOLE))

initial_logging_setup()

logger = logging.getLogger(__name__) # For app.py ('__main__')
logger.info("--- app.py module logger initialized ---")

# ... (rest of your app.py imports and code remain IDENTICAL to the last version I provided that fixed the f-string error) ...
# Standard library imports
import hashlib
# import warnings # Already imported above
import tempfile
import json
import base64
import io
import traceback
from datetime import datetime
import uuid

# Third-party library imports
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import psutil
import torch
import numpy as np
import torchaudio
from flask import Flask, request, jsonify, make_response, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

logger.info("--- app.py: Starting service module imports ---")
from services.audio_link_routes import handle_audio_url_processing
from services.audio_processor import AudioProcessor
from services.model_manager import ModelManager, DEVICE as APP_WIDE_DEVICE
from services.error_handler import ErrorHandler
from services.utils import cleanup_file, performance_logger
from services.video_routes import handle_video_processing
from services.podcast_routes import handle_podcast_upload
from services.health_routes import handle_model_health
from services.translation_strategy import TranslationManager
from services.cascaded_backend import CascadedBackend, check_openvoice_api
logger.info("--- app.py: Core service module imports complete ---")

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

MAX_AUDIO_LENGTH = 300
MAX_PODCAST_LENGTH = 3600
SAMPLE_RATE = 16000
UPLOAD_FOLDER = Path('uploads/podcasts')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

logger.info("Creating ModelManager instance..."); model_manager_instance = ModelManager()
logger.info("Creating TranslationManager instance..."); translation_manager = TranslationManager()

hf_auth_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_auth_token:
    logger.warning("HUGGINGFACE_TOKEN environment variable not set. Model downloads may fail.")

logger.info("Attempting to instantiate and register default backend (CascadedBackend)...")
try:
    cascaded_backend_instance = CascadedBackend(device=APP_WIDE_DEVICE, use_voice_cloning=True)
    translation_manager.register_backend("cascaded", cascaded_backend_instance, is_default=True)
except Exception as e_backend_reg:
    logger.critical(f"CRITICAL FAILURE: Could not instantiate/register CascadedBackend: {e_backend_reg}", exc_info=True)
    sys.exit(1)

app = Flask(__name__); logger.info("Flask app instance created.")
CORS(app, resources={ r"/*": { "origins": ["http://localhost:3000", "http://localhost:3001"], "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Accept", "Authorization", "X-Requested-With", "Range", "Accept-Ranges", "Origin"], "expose_headers": ["Content-Type", "Content-Length", "Content-Range", "Content-Disposition", "Accept-Ranges", "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"], "supports_credentials": True, "max_age": 120, "automatic_options": True }})
logger.info("CORS configured.")
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
logger.info("Flask-Limiter initialized.")
app.config['IS_SHUTTING_DOWN'] = False
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024

@app.before_request
def before_request_middleware():
    if request.method == 'OPTIONS': return
    if request.method not in ['GET', 'POST', 'HEAD']:
        logger.warning(f"Method not allowed: {request.method} for {request.path}")
        return jsonify({'error': 'Method not allowed'}), 405
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type', '').lower()
        if not content_type.startswith('multipart/form-data') and \
           not content_type.startswith('application/json'):
            if not (request.form or request.files):
                 logger.warning(f"Invalid content type for POST: '{content_type}' for {request.path}. Expected JSON or multipart/form-data.")
                 return jsonify({'error': 'Invalid content type for POST. Expected JSON or multipart/form-data.'}), 400
    request.start_time = time.time()

@app.after_request
def after_request_middleware(response: Response) -> Response:
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"REQ: {request.method} {request.path} ({request.remote_addr}) -> RSP: {response.status_code} [{duration:.2f}s]")
    else:
        logger.info(f"REQ: {request.method} {request.path} ({request.remote_addr}) -> RSP: {response.status_code} [duration N/A]")
    return response

@app.errorhandler(Exception)
def central_error_handler(e: Exception):
    error_response, status_code = ErrorHandler.handle_error(e); return error_response, status_code

# --- Routes ---
@app.route('/translate', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
@performance_logger
def translate_audio_route():
    if request.method == 'OPTIONS': resp = make_response(); return resp
    temp_files_to_clean = []
    try:
        backend_to_use = translation_manager.get_backend()
        logger.info(f"Using backend: {type(backend_to_use).__name__} for /translate")
        if 'file' not in request.files: return ErrorHandler.format_validation_error('No file uploaded')
        file = request.files['file']
        if not file.filename: return ErrorHandler.format_validation_error('No file selected')
        target_language_app_code = request.form.get('target_language', 'fra')
        logger.info(f"/translate: Target='{target_language_app_code}' for file '{file.filename}'")

        audio_processor = AudioProcessor()
        temp_suffix = Path(secure_filename(file.filename)).suffix or '.tmp'
        temp_dir_uploads = Path(tempfile.gettempdir()) / "magenta_translate_uploads"
        temp_dir_uploads.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix, dir=temp_dir_uploads) as tmp_in_file:
            file.save(tmp_in_file.name)
            temp_files_to_clean.append(tmp_in_file.name)
            logger.debug(f"Uploaded file saved to temp path: {tmp_in_file.name}")
        
        valid_audio, error_msg = audio_processor.validate_audio_length(tmp_in_file.name)
        if not valid_audio:
             logger.error(f"Audio validation failed for {tmp_in_file.name}: {error_msg}")
             return ErrorHandler.format_validation_error(error_msg)
        
        audio_tensor = audio_processor.process_audio(tmp_in_file.name)
        if not audio_processor.is_valid_audio(audio_tensor):
            logger.error(f"Processed audio tensor is invalid for {tmp_in_file.name}")
            return ErrorHandler.handle_error(ValueError("Processed audio is invalid or silent."))

        result = backend_to_use.translate_speech(audio_tensor=audio_tensor, source_lang="eng", target_lang=target_language_app_code)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir_uploads) as tmp_out_file:
            temp_files_to_clean.append(tmp_out_file.name)
            audio_save = result["audio"]
            if audio_save.ndim > 1 and audio_save.shape[0] == 1: audio_save = audio_save.squeeze(0)
            torchaudio.save(tmp_out_file.name, audio_save.cpu(), SAMPLE_RATE)
            logger.debug(f"Translated audio saved to temp path: {tmp_out_file.name}")
            with open(tmp_out_file.name, 'rb') as audio_data_obj:
                audio_b64 = base64.b64encode(audio_data_obj.read()).decode('utf-8')
        return jsonify({'audio': audio_b64, 'transcripts': result["transcripts"]})
    except Exception as e:
        logger.error(f"Error in /translate route: {e}", exc_info=True)
        return ErrorHandler.handle_error(e)
    finally:
        for f_path in temp_files_to_clean: cleanup_file(f_path)

@app.route('/process-video', methods=['POST', 'OPTIONS'])
@limiter.limit("5 per minute")
@performance_logger
def process_video_route():
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
        logger.info(f"Video processing: Target Lang='{target_language}', Apply Lip Sync='{apply_lip_sync_bool}'")
        video_processing_backend = translation_manager.get_backend('cascaded')
        if not video_processing_backend.initialized:
            logger.error("Failed to get/initialize CascadedBackend for video in /process-video.")
            def err_stream_backend(): yield f"data: {json.dumps({'error': 'Backend service not ready.', 'phase': 'Backend Error'})}\n\n"
            return Response(stream_with_context(err_stream_backend()), mimetype='text/event-stream', status=503)
        if hasattr(video_processing_backend, 'use_voice_cloning_config'):
            form_voice_cloning = request.form.get('use_voice_cloning', 'true').lower() == 'true'
            video_processing_backend.use_voice_cloning_config = form_voice_cloning
            logger.info(f"Video processing: Set use_voice_cloning on '{type(video_processing_backend).__name__}' to: {form_voice_cloning}")
        else:
            logger.warning(f"Backend '{type(video_processing_backend).__name__}' does not have 'use_voice_cloning_config' attribute.")
        return handle_video_processing(request.files, target_language, video_processing_backend, apply_lip_sync_bool)
    except Exception as e:
        logger.error(f"Error in /process-video route: {e}", exc_info=True)
        def err_stream_exc():
            err_payload = json.dumps({'error': 'Internal server error during video processing.','phase': 'Server Error','details': str(e)[:200]})
            yield f"data: {err_payload}\n\n"
        return Response(stream_with_context(err_stream_exc()), mimetype='text/event-stream', status=500)

@app.route('/available-backends', methods=['GET'])
def available_backends_route():
    try:
        backends_dict = translation_manager.get_available_backends(); backend_names = list(backends_dict.keys())
        default_backend_name = translation_manager.default_backend
        return jsonify({'backends': backend_names, 'default': default_backend_name})
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/supported-languages', methods=['GET'])
def supported_languages_route():
    try:
        backend_name_query = request.args.get('backend')
        backend_to_check = translation_manager.get_backend(backend_name_query)
        return jsonify({'languages': backend_to_check.get_supported_languages()})
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/process-audio-url', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
@performance_logger
def process_audio_url_route():
    if request.method == 'OPTIONS': resp = make_response(); return resp
    try:
        data = request.get_json();
        if not data or 'url' not in data: return ErrorHandler.format_validation_error('No URL provided')
        url = data['url']
        result = handle_audio_url_processing(url)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int): return jsonify(result[0]), result[1]
        if 'error' in result and isinstance(result, dict) : return jsonify(result), result.get('status_code', 400)
        response = make_response(result['audio_data']); response.headers['Content-Type'] = result['mime_type']
        return response
    except Exception as e: return ErrorHandler.handle_error(e)


@app.route('/openvoice-status', methods=['GET'])
def openvoice_status_route():
    try:
        is_available = check_openvoice_api()
        return jsonify({'available': is_available, 'message': "OpenVoice API available and models loaded." if is_available else "OpenVoice API not available or models not loaded."})
    except Exception as e:
        logger.error(f"Error checking OpenVoice status: {e}", exc_info=True)
        return jsonify({'available': False, 'message': f"Error checking OpenVoice: {str(e)}"}), 500

@app.route('/upload_podcast', methods=['POST', 'OPTIONS'])
@limiter.limit("5 per minute")
@performance_logger
def upload_podcast_route():
    if request.method == 'OPTIONS': resp = make_response(); return resp
    return handle_podcast_upload(UPLOAD_FOLDER, MAX_PODCAST_LENGTH, ALLOWED_EXTENSIONS)

@app.route('/health/model', methods=['GET'])
def model_health_route():
    return handle_model_health(APP_WIDE_DEVICE, translation_manager, model_manager_instance)


# --- Graceful Shutdown Logic ---
def _app_cleanup_tasks():
    logger.info("Shutdown: Cleaning up application resources...")
    try:
        if model_manager_instance and hasattr(model_manager_instance, '_models_loaded') and model_manager_instance._models_loaded:
            logger.info("Cleaning up ModelManager (SeamlessM4T)..."); model_manager_instance.cleanup()
        else: logger.info("ModelManager (for optional on-demand models like SeamlessM4T) not loaded or already cleaned, skipping its cleanup.")
        if 'translation_manager' in globals() and translation_manager.backends:
            logger.info("Cleaning up translation backends...")
            for name, backend in translation_manager.get_available_backends().items():
                if hasattr(backend, 'cleanup') and callable(backend.cleanup):
                    logger.info(f"Calling .cleanup() for backend: {name}");
                    try: backend.cleanup()
                    except Exception as e_clean: logger.error(f"Error cleaning backend {name}: {e_clean}", exc_info=True)
        else: logger.info("TranslationManager or its backends not found, skipping backend cleanup.")
        if torch.cuda.is_available(): logger.info("Emptying CUDA cache..."); torch.cuda.empty_cache()
        gc.collect(); logger.info("Resource cleanup by _app_cleanup_tasks complete.")
    except Exception as e: logger.error(f"Error during _app_cleanup_tasks: {e}", exc_info=True)

def _graceful_shutdown_signal_handler(_signum, _frame):
    logger.info(f"Signal {_signum} received, initiating graceful shutdown...")
    if not app.config.get('IS_SHUTTING_DOWN', False):
        app.config['IS_SHUTTING_DOWN'] = True
        logger.info("Graceful shutdown tasks to be run by atexit handler. Exiting now to trigger atexit.")
        sys.exit(0)

atexit.register(_app_cleanup_tasks)
signal.signal(signal.SIGTERM, _graceful_shutdown_signal_handler)
signal.signal(signal.SIGINT, _graceful_shutdown_signal_handler)
# --- End Graceful Shutdown Logic ---

if __name__ == '__main__':
    try:
        logger.info("="*50);
        logger.info("Starting Magenta AI Backend Main Application (app.py)")
        logger.info(f"Running with Python version: {sys.version.splitlines()[0]}")
        if not os.getenv('HUGGINGFACE_TOKEN'): logger.warning("HUGGINGFACE_TOKEN environment variable not set. Some model downloads may fail.")
        logger.info(f"Attempting to initialize default S2ST backend: '{translation_manager.default_backend or 'Cascaded (default will be set)'}'")
        try:
            default_backend_instance = translation_manager.get_backend()
            if not default_backend_instance.initialized:
                 logger.critical(f"CRITICAL FAILURE: Default backend '{translation_manager.default_backend}' DID NOT successfully initialize AFTER get_backend call. Check logs from its initialization routine (e.g., CascadedBackend).")
                 sys.exit(1)
            logger.info(f"Default S2ST backend '{translation_manager.default_backend}' (type: {type(default_backend_instance).__name__}) confirmed initialized.")
        except Exception as e_init_check:
            logger.critical(f"CRITICAL FAILURE during default backend get/initialization check: {e_init_check}", exc_info=True)
            sys.exit(1)
        logger.info("ModelManager (for optional on-demand models like SeamlessM4T) instance confirmed created.")
        logger.info(f"Flask app starting on port 5001. ML Processing Device: {APP_WIDE_DEVICE}")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)
    except SystemExit:
        logger.info("Application exiting due to SystemExit (likely from signal handler).")
    except Exception as e_startup:
        logger.critical(f"FATAL: Application startup sequence failed: {e_startup}", exc_info=True)
    finally:
        logger.info("Application shutdown sequence initiated or completed.")