from .audio_processor import AudioProcessor
# from .model_manager import ModelManager # Removed as ModelManager.py is being deleted
from .resource_monitor import ResourceMonitor # Keep if still used (e.g., by podcast_routes)
from .error_handler import ErrorHandler
from .utils import (
    cleanup_file,
    # require_model, # Removed as it depended on the old ModelManager
    performance_logger
)
# Ensure all other relevant exports from utils.py or other service modules are here if needed

__all__ = [
    'AudioProcessor',
    # 'ModelManager', # Removed
    'ResourceMonitor', # Keep if still used
    'ErrorHandler',
    'cleanup_file',
    # 'require_model', # Removed
    'performance_logger'
    # Add other classes/functions from your services modules that you want to be easily importable
    # e.g., from .cascaded_backend import CascadedBackend
    # from .translation_strategy import TranslationManager
    # from .video_routes import handle_video_processing
    # ... etc.
    # It's good practice to list all public interfaces of your 'services' package here.
]

# For example, if you want to be able to do `from services import CascadedBackend`:
from .cascaded_backend import CascadedBackend
from .translation_strategy import TranslationManager
# ... add other important classes/functions ...

# Then update __all__ accordingly:
# __all__.extend(['CascadedBackend', 'TranslationManager'])
# This part is optional but good practice for package interfaces.
# For now, I'll stick to modifying based on deletions.