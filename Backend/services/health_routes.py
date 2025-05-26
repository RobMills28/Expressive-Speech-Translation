from flask import jsonify
import psutil
import torch
from datetime import datetime
import logging

# Removed: from .model_manager import ModelManager
# If TranslationManager type hint is desired for clarity:
# from .translation_strategy import TranslationManager

logger = logging.getLogger(__name__)

# The function signature now matches the call from app.py
def handle_model_health(app_wide_device, translation_manager_instance, model_manager_instance_arg): # model_manager_instance_arg will be None
    """Detailed model health check"""
    try:
        health_info = {
            'status': 'healthy',
            'primary_device': str(app_wide_device), # Use the passed device info
            'timestamp': datetime.now().isoformat()
        }

        # Default Backend Info from TranslationManager
        if translation_manager_instance:
            try:
                # Get the default backend, which should also initialize it if not already
                default_backend = translation_manager_instance.get_backend()
                health_info['default_backend_type'] = type(default_backend).__name__
                health_info['default_backend_initialized'] = default_backend.initialized
            except Exception as e_backend:
                logger.error(f"Error getting default backend info for health check: {e_backend}", exc_info=True)
                health_info['default_backend_status'] = f'Error retrieving default backend: {str(e_backend)}'
        else:
            health_info['default_backend_status'] = 'TranslationManager instance not provided to health check.'


        # Information about the (now removed) ModelManager / SeamlessM4T
        # model_manager_instance_arg will be None, so this reflects its removal.
        if model_manager_instance_arg:
            # This block would only be entered if ModelManager was NOT removed,
            # which is contrary to the current plan.
            # For completeness if ModelManager had other roles:
            if hasattr(model_manager_instance_arg, '_models_loaded') and model_manager_instance_arg._models_loaded:
                 health_info['optional_on_demand_model_status'] = 'loaded (ModelManager was active)'
            else:
                 health_info['optional_on_demand_model_status'] = 'not_loaded (ModelManager was active but models not loaded)'
        else:
            health_info['optional_on_demand_model_status'] = 'manager_not_active (ModelManager for SeamlessM4T removed)'


        # System Memory
        process = psutil.Process(os.getpid()) # Get current process
        health_info['memory_usage'] = {
            'system_rss_mb': process.memory_info().rss / (1024**2),
            'system_virtual_percent': psutil.virtual_memory().percent,
        }

        # GPU Memory if CUDA device
        if app_wide_device and app_wide_device.type == 'cuda' and torch.cuda.is_available():
            try:
                gpu_idx = app_wide_device.index if app_wide_device.index is not None else torch.cuda.current_device()
                health_info['memory_usage']['gpu_mb'] = {
                    'allocated': torch.cuda.memory_allocated(gpu_idx) / (1024**2),
                    'reserved_cached': torch.cuda.memory_reserved(gpu_idx) / (1024**2), # More descriptive term
                    'total_capacity': torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**2)
                }
            except Exception as e_gpu:
                logger.error(f"Error getting GPU memory info: {e_gpu}", exc_info=True)
                health_info['memory_usage']['gpu_mb'] = "Error retrieving GPU memory"
        elif app_wide_device and app_wide_device.type == 'cuda' and not torch.cuda.is_available():
            health_info['memory_usage']['gpu_status'] = "CUDA device specified but torch.cuda.is_available() is False"


        return jsonify(health_info)

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500