# app.py

# Standard library imports
import os
import sys
import time
import atexit
import signal
import logging 
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path 

# --- Call Logging Setup VERY EARLY ---
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
    root_logger.setLevel(logging.INFO)

    main_handler = TimedRotatingFileHandler(log_dir / 'app.log', when='midnight', interval=1, backupCount=30, encoding='utf-8')
    main_handler.setFormatter(detailed_formatter); main_handler.setLevel(logging.INFO)
    root_logger.addHandler(main_handler)
    
    error_handler = RotatingFileHandler(log_dir / 'error.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    error_handler.setLevel(logging.ERROR); error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter); console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('pydub.utils').setLevel(logging.WARNING)
    logging.info("Root logging configured at app startup.")

setup_logging() 
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

import hashlib
import warnings
import tempfile
import json
import base64
import io
import traceback
from datetime import datetime
import uuid
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

from services.audio_link_routes import handle_audio_url_processing
from services.audio_processor import AudioProcessor
from services.model_manager import ModelManager, DEVICE as APP_WIDE_DEVICE 
from services.resource_monitor import ResourceMonitor
from services.error_handler import ErrorHandler
from services.utils import cleanup_file, performance_logger
from services.video_routes import handle_video_processing 
from services.podcast_routes import handle_podcast_upload
from services.health_routes import handle_model_health 

from services.translation_strategy import TranslationManager, TranslationBackend
from services.seamless_backend import SeamlessBackend 
from services.cascaded_backend import CascadedBackend, check_openvoice_api 

try:
    from services.espnet_backend import ESPnetBackend
    ESPNET_AVAILABLE = True
    logger.info("ESPnet backend components potentially available.")
except ImportError:
    logger.warning("ESPnet native library not found or import failed, ESPnetBackend will not be available.")
    ESPNET_AVAILABLE = False

load_dotenv()

MAX_AUDIO_LENGTH = 300 
MAX_PODCAST_LENGTH = 3600  
SAMPLE_RATE = 16000

UPLOAD_FOLDER = Path('uploads/podcasts')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

model_manager_instance = ModelManager() 
logger.info("ModelManager (for optional SeamlessM4T) instance created. Models will load on demand if routes use it.")

app = Flask(__name__)
CORS(app, resources={ r"/*": { 
    "origins": ["http://localhost:3000"], "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Accept", "Authorization", "X-Requested-With", "Range", "Accept-Ranges", "Origin"],
    "expose_headers": ["Content-Type", "Content-Length", "Content-Range", "Content-Disposition", "Accept-Ranges", "Access-Control-Allow-Origin", "Access-Control-Allow-Credentials"],
    "supports_credentials": True, "max_age": 120, "automatic_options": True }})
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

translation_manager = TranslationManager()
hf_auth_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_auth_token: logger.warning("HUGGINGFACE_TOKEN not set. Some model downloads (NLLB, Whisper from HF) might fail.")

try:
    cascaded_backend = CascadedBackend(device=APP_WIDE_DEVICE, use_voice_cloning=True) 
    translation_manager.register_backend("cascaded", cascaded_backend, is_default=True)
except Exception as e:
    logger.critical(f"CRITICAL: Failed to instantiate/register default CascadedBackend: {e}", exc_info=True)

app.config['IS_SHUTTING_DOWN'] = False
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024 

@app.before_request
def before_request_funcs():
    if request.method not in ['GET', 'POST', 'OPTIONS', 'HEAD']: return jsonify({'error': 'Method not allowed'}), 405
    if request.method == 'POST' and not request.content_type.startswith('multipart/form-data') and not request.is_json:
        if not (request.files or request.form): return jsonify({'error': 'Invalid content type'}), 400
    request.start_time = time.time()

@app.after_request
def after_request_funcs(response):
    origin = request.headers.get('Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"REQ: {request.method} {request.path} ({request.remote_addr}) -> {response.status_code} [{duration:.2f}s]")
    return response

@app.errorhandler(Exception) 
def handle_central_error_handler(e): return ErrorHandler.handle_error(e)

@app.route('/translate', methods=['POST', 'OPTIONS'])
@performance_logger # Apply this first
@limiter.limit("10 per minute") 
def translate_audio_endpoint_cascaded(): 
    logger.info("Request for /translate (audio-only, using default CascadedBackend).")
    temp_files_to_clean = []
    try:
        backend_to_use = translation_manager.get_backend() 
        if not backend_to_use:
            return jsonify({'error': 'Translation service (default) unavailable.'}), 503

        if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file.filename: return jsonify({'error': 'No file selected'}), 400
        
        target_language_app_code = request.form.get('target_language', 'fra') 
        logger.info(f"/translate: Target='{target_language_app_code}', Backend='{type(backend_to_use).__name__}'")

        audio_processor = AudioProcessor()
        temp_suffix = Path(secure_filename(file.filename)).suffix or '.tmp'
        temp_dir_for_uploads = Path(tempfile.gettempdir()) / "magenta_uploads"
        temp_dir_for_uploads.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix, dir=temp_dir_for_uploads) as temp_input_file:
            file.save(temp_input_file.name)
            temp_files_to_clean.append(temp_input_file.name)
        
        audio_tensor = audio_processor.process_audio(temp_input_file.name) 
        
        result = backend_to_use.translate_speech(
            audio_tensor=audio_tensor, source_lang="eng", target_lang=target_language_app_code
        )
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir_for_uploads) as temp_output_audio_file:
            temp_files_to_clean.append(temp_output_audio_file.name)
            audio_to_save = result["audio"]
            if audio_to_save.ndim > 1 and audio_to_save.shape[0] == 1: audio_to_save = audio_to_save.squeeze(0)
            torchaudio.save(temp_output_audio_file.name, audio_to_save.cpu(), SAMPLE_RATE)
            with open(temp_output_audio_file.name, 'rb') as audio_file_data_obj:
                audio_data_b64 = base64.b64encode(audio_file_data_obj.read()).decode('utf-8')
        
        return jsonify({'audio': audio_data_b64, 'transcripts': result["transcripts"]})
    except Exception as e: return ErrorHandler.handle_error(e)
    finally:
        for f_path in temp_files_to_clean: cleanup_file(f_path)

@app.route('/available-backends', methods=['GET'])
def available_backends_endpoint():
    try:
        backends_dict = translation_manager.get_available_backends(); backend_names = list(backends_dict.keys())
        default_backend_name = translation_manager.default_backend
        return jsonify({'backends': backend_names, 'default': default_backend_name})
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/supported-languages', methods=['GET'])
def supported_languages_endpoint():
    try:
        backend_name = request.args.get('backend')
        backend_to_check = translation_manager.get_backend(backend_name)
        if not backend_to_check: return jsonify({'error': f"Backend '{backend_name or 'default'}' not found."}), 404
        return jsonify({'languages': backend_to_check.get_supported_languages()})
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/process-audio-url', methods=['POST', 'OPTIONS'])
@performance_logger # Apply this first
@limiter.limit("10 per minute")
def process_audio_url_endpoint():
    if request.method == 'OPTIONS': return make_response() 
    try:
        data = request.get_json();
        if not data or 'url' not in data: return jsonify({'error': 'No URL provided'}), 400
        url = data['url']; result = handle_audio_url_processing(url) 
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int): return jsonify(result[0]), result[1]
        if 'error' in result: return jsonify(result), 400
        response = make_response(result['audio_data']); response.headers['Content-Type'] = result['mime_type']
        return response
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/process-video', methods=['POST', 'OPTIONS'])
@performance_logger # Apply this first
@limiter.limit("5 per minute") 
def process_video_route_handler():
    logger.info("Request for /process-video")
    if request.method == 'OPTIONS': return make_response() 
    try:
        target_language = request.form.get('target_language', 'fra') 
        backend_name_to_use = 'cascaded' 
        logger.info(f"Video processing: Using backend '{backend_name_to_use}' for lang '{target_language}'")
        
        video_processing_backend = translation_manager.get_backend(backend_name_to_use)
        if not video_processing_backend:
            return jsonify({'error': f"Video backend '{backend_name_to_use}' unavailable."}), 503

        if hasattr(video_processing_backend, 'use_voice_cloning_config'):
            form_voice_cloning = request.form.get('use_voice_cloning', 'true').lower() == 'true'
            video_processing_backend.use_voice_cloning_config = form_voice_cloning
            logger.info(f"Video processing: Set use_voice_cloning on '{backend_name_to_use}' to: {form_voice_cloning}")
        
        return handle_video_processing(request.files, target_language, video_processing_backend)
    except Exception as e: return ErrorHandler.handle_error(e)

@app.route('/openvoice-status', methods=['GET'])
def openvoice_status_endpoint():
    try:
        is_available = check_openvoice_api() 
        return jsonify({'available': is_available, 'message': "OpenVoice API available" if is_available else "OpenVoice API not available"})
    except Exception as e: return jsonify({'available': False, 'message': f"Error: {str(e)}"}), 500

@app.route('/upload_podcast', methods=['POST', 'OPTIONS'])
@performance_logger # Apply this first
@limiter.limit("5 per minute") 
def upload_podcast_endpoint():
    return handle_podcast_upload(UPLOAD_FOLDER, MAX_PODCAST_LENGTH, ALLOWED_EXTENSIONS)
            
@app.route('/health/model', methods=['GET'])
@performance_logger 
def model_health_endpoint():
    health_info = {'status': 'healthy', 'device': str(APP_WIDE_DEVICE)}
    try:
        default_backend = translation_manager.get_backend() 
        health_info['default_backend_status'] = 'initialized' if default_backend.initialized else 'not_initialized'
        health_info['default_backend_type'] = type(default_backend).__name__
        if model_manager_instance._models_loaded: 
             health_info['optional_seamless_m4t_status'] = 'loaded'
        else:
             health_info['optional_seamless_m4t_status'] = 'not_loaded_or_not_used'
    except Exception as e:
        health_info['default_backend_status'] = f'error_retrieving ({e})'
    return jsonify(health_info)

def graceful_shutdown_signal_handler(_signum, _frame): logger.info(f"Signal {_signum} received, initiating graceful shutdown..."); graceful_shutdown()
def graceful_shutdown():
    logger.info("Graceful shutdown: Setting shutdown flag..."); app.config['IS_SHUTTING_DOWN'] = True
    time.sleep(1); shutdown_handler_tasks(); logger.info("Graceful shutdown: Tasks completed.")

def shutdown_handler_tasks(): 
    logger.info("Shutdown handler: Cleaning up resources...")
    try:
        if model_manager_instance and hasattr(model_manager_instance, '_models_loaded') and model_manager_instance._models_loaded: 
            logger.info("Cleaning up ModelManager (SeamlessM4T components)..."); model_manager_instance.cleanup()
        
        if 'translation_manager' in globals():
            for backend_name, backend_inst in translation_manager.get_available_backends().items():
                if hasattr(backend_inst, 'cleanup') and callable(backend_inst.cleanup):
                    logger.info(f"Cleaning up backend: {backend_name}"); 
                    try: backend_inst.cleanup()
                    except Exception as e: logger.error(f"Error cleaning {backend_name}: {e}")
                elif hasattr(backend_inst, 'melo_tts_models_cache') and backend_inst.melo_tts_models_cache: 
                    logger.info(f"Releasing MeloTTS models for backend: {backend_name}")
                    del backend_inst.melo_tts_models_cache; backend_inst.melo_tts_models_cache = {}
                
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect(); logger.info("All resource cleanup attempts completed.")
    except Exception as e: logger.error(f"Error during shutdown_handler_tasks: {e}", exc_info=True)

atexit.register(shutdown_handler_tasks)
signal.signal(signal.SIGTERM, graceful_shutdown_signal_handler)
signal.signal(signal.SIGINT, graceful_shutdown_signal_handler)

if __name__ == '__main__':
    try:
        logger.info("="*50); logger.info("Starting Magenta AI Backend")
        
        if not os.getenv('HUGGINGFACE_TOKEN'):
            logger.critical("CRITICAL: HUGGINGFACE_TOKEN not set. Model downloads will likely fail.")
        
        logger.info(f"Attempting to initialize default translation backend: '{translation_manager.default_backend or 'Not Set Yet'}'")
        if translation_manager.default_backend:
            try:
                default_backend = translation_manager.get_backend() 
                if not default_backend or not (hasattr(default_backend, 'initialized') and default_backend.initialized):
                    raise RuntimeError(f"Default backend '{translation_manager.default_backend}' failed to initialize properly.")
                logger.info(f"Default backend '{translation_manager.default_backend}' (type: {type(default_backend).__name__}) initialized successfully.")
            except Exception as e_init_backend:
                logger.critical(f"CRITICAL FAILURE: Could not initialize default backend '{translation_manager.default_backend}': {e_init_backend}", exc_info=True)
                sys.exit(1)
        else:
            logger.critical("CRITICAL FAILURE: No default backend configured in TranslationManager after setup."); sys.exit(1)
        
        logger.info("ModelManager (for optional SeamlessM4T) created; models load on-demand if specific routes call get_model_components().")
        logger.info(f"Flask app starting on port 5001 using device for ML: {APP_WIDE_DEVICE}")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)
        
    except Exception as e_startup:
        logger.critical(f"FATAL: Application startup failed: {e_startup}", exc_info=True)
        try: shutdown_handler_tasks()
        except: pass
        sys.exit(1)