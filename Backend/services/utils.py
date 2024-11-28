import os
import time
import json
import psutil
import logging
import hashlib
from functools import wraps
from flask import jsonify
from .model_manager import ModelManager

# Set up logging
logger = logging.getLogger(__name__)

def generate_progress_event(progress: int, status: str) -> str:
    """Generate a Server-Sent Event for progress updates"""
    return f"data: {json.dumps({'progress': progress, 'status': status})}\n\n"

def cleanup_file(file_path: str) -> None:
    """Clean up a temporary file"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

def require_model(f):
    """Decorator to ensure model is loaded before processing"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        model_manager = ModelManager()
        processor, model, tokenizer = model_manager.get_model_components()
        
        if None in (processor, model, tokenizer):
            return jsonify({'error': 'Model not initialized properly'}), 503
            
        return f(*args, **kwargs)
    return decorated_function

def performance_logger(f):
    """Decorator to log function performance metrics"""
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
            
            logger.info(
                f"Performance - Function: {f.__name__}, Duration: {duration:.2f}s, "
                f"Memory Change: {memory_diff/1024/1024:.2f}MB"
            )
            
            return result
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            raise
            
    return decorated_function