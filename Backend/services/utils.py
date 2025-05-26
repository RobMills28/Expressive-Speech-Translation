import os
import time
import json
import psutil
import logging
import hashlib
from functools import wraps
from flask import jsonify
# Removed: from .model_manager import ModelManager # This was causing the ModuleNotFoundError

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

# The 'require_model' decorator is removed as its dependency 'ModelManager' has been deleted.
# If you have other global models that need a similar decorator,
# you would need a new mechanism to access and check them.
# For now, since SeamlessM4T was the target, this decorator is removed.
#
# def require_model(f):
#     """Decorator to ensure model is loaded before processing"""
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         global processor, model, text_model, tokenizer # These globals were tied to ModelManager
#
#         # This part is no longer valid as ModelManager is removed.
#         # model_manager = ModelManager()
#         # processor, model, text_model, tokenizer = model_manager.get_model_components()
#
#         # if None in (processor, model, text_model, tokenizer):
#         #     return jsonify({'error': 'Model not initialized properly'}), 503
#
#         # Placeholder if you need to re-implement for other models:
#         # Check if your primary models (e.g., from CascadedBackend) are ready
#         # This would require passing the translation_manager or backend instance
#         # or having another way to check their status.
#         # For simplicity, this check is removed. If needed, it has to be re-thought.
#
#         return f(*args, **kwargs)
#     return decorated_function

def performance_logger(f):
    """Decorator to log function performance metrics"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        # Get current process's memory usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss

        try:
            result = f(*args, **kwargs)

            end_time = time.time()
            end_memory = process.memory_info().rss
            duration = end_time - start_time
            memory_diff = end_memory - start_memory

            logger.info(
                f"Performance - Function: {f.__name__}, Duration: {duration:.2f}s, "
                f"Memory Change: {memory_diff/1024/1024:.2f}MB"
            )

            return result
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True) # Added exc_info for better debugging
            raise # Re-raise the exception after logging

    return decorated_function