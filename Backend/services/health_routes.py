from flask import jsonify
import psutil
import torch
from datetime import datetime
import logging

from .model_manager import ModelManager

logger = logging.getLogger(__name__)

def handle_model_health(DEVICE):
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