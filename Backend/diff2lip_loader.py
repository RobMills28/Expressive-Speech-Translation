# Save this as diff2lip_loader.py in your Backend directory

import io
import os
import sys
import torch
import pickle
import logging

logger = logging.getLogger(__name__)

def custom_load_model(model_path, map_location='cpu'):
    """
    Custom function to load Diff2Lip model that handles persistent IDs.
    """
    try:
        logger.info(f"Attempting to load model from {model_path} with custom loader")
        
        # Read the model file
        with open(model_path, 'rb') as f:
            data = f.read()
            
        # Define a custom unpickler class
        class CustomUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                # This is the key - just return the ID instead of trying to load it
                return pid
                
        # Try with custom unpickler
        buffer = io.BytesIO(data)
        unpickler = CustomUnpickler(buffer)
        model_dict = unpickler.load()
        
        logger.info("Successfully loaded model with custom unpickler")
        return model_dict
        
    except Exception as e:
        logger.error(f"Custom model loading failed: {str(e)}")
        # Fall back to simple torch.load as a last resort
        return torch.load(model_path, map_location=map_location)