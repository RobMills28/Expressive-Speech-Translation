from flask import request, jsonify
import hashlib
import time
import torchaudio
from pathlib import Path
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

from .resource_monitor import ResourceMonitor
from .error_handler import ErrorHandler
from .utils import cleanup_file

logger = logging.getLogger(__name__)

def handle_podcast_upload(UPLOAD_FOLDER, MAX_PODCAST_LENGTH, ALLOWED_EXTENSIONS):
    if request.method == 'OPTIONS':
        return '', 204
        
    request_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    start_time = time.time()
    
    try:
        logger.info(f"Starting podcast upload request {request_id}")
        
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
            
        if not '.' in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            logger.error(f"Request {request_id}: Invalid file type")
            return jsonify({'error': 'Invalid file type. Allowed types: mp3, wav, ogg, m4a'}), 400

        # Generate unique filename and save file
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        unique_filename = f"{unique_id}_{filename}"
        filepath = UPLOAD_FOLDER / unique_filename

        # Save and process file
        try:
            file.save(filepath)
            logger.info(f"Request {request_id}: Saved file to {filepath}")
            
            # Get audio duration using torchaudio
            waveform, sample_rate = torchaudio.load(filepath)
            duration_seconds = waveform.shape[1] / sample_rate
            
            # Check if podcast exceeds maximum length
            if duration_seconds > MAX_PODCAST_LENGTH:
                cleanup_file(filepath)
                return jsonify({'error': f'Podcast exceeds maximum length of {MAX_PODCAST_LENGTH/60} minutes'}), 400
            
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration = f"{minutes:02d}:{seconds:02d}"
            
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            cleanup_file(filepath)
            return jsonify({'error': 'Failed to process audio file'}), 400

        # Create response object
        podcast_data = {
            'id': unique_id,
            'title': request.form.get('title', file.filename.rsplit('.', 1)[0]),
            'episode': str(len(list(UPLOAD_FOLDER.glob('*')))), 
            'duration': duration,
            'date': datetime.now().isoformat(),
            'filepath': str(filepath)
        }
        
        logger.info(f"Request {request_id}: Successfully processed podcast upload: {podcast_data['title']}")
        return jsonify(podcast_data), 200
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unhandled error: {str(e)}", exc_info=True)
        return ErrorHandler.handle_error(e)
        
    finally:
        duration = time.time() - start_time
        logger.info(f"Request {request_id} completed in {duration:.2f}s")