#!/usr/bin/env python3
"""
Speaker Embedding Creator for OpenVoice

This script creates a compatible speaker embedding file for OpenVoice
from an existing embedding file, ensuring it has the correct shape and format.
"""

import os
import torch
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate a compatible en-us.pth file for OpenVoice")
    parser.add_argument("--source_file", type=str, default="en-us.pth", 
                       help="Original en-us.pth file to use as reference")
    parser.add_argument("--output_file", type=str, default="en-us-new.pth", 
                       help="Path to save the new embedding file")
    args = parser.parse_args()
    
    # Define paths
    source_path = Path(args.source_file)
    output_path = Path(args.output_file)
    
    print(f"Creating a new compatible speaker embedding file")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    try:
        # Create compatible tensor data
        # First try loading directly (different PyTorch versions may need different approaches)
        embedding = None
        success = False
        
        # Try multiple loading methods to handle different PyTorch versions
        loading_methods = [
            {"method": "default", "kwargs": {}},
            {"method": "weights_only_true", "kwargs": {"weights_only": True}},
            {"method": "weights_only_false", "kwargs": {"weights_only": False}},
            {"method": "map_location", "kwargs": {"map_location": "cpu"}},
            {"method": "pickle_safe", "kwargs": {"pickle_module": torch.serialization._pickle}}
        ]
        
        for method in loading_methods:
            try:
                logger.info(f"Trying to load with {method['method']}")
                embedding = torch.load(source_path, **method["kwargs"])
                logger.info(f"Successfully loaded with {method['method']}")
                success = True
                break
            except Exception as e:
                logger.warning(f"Failed to load with {method['method']}: {str(e)}")
        
        # If all loading attempts failed, create a random embedding
        if not success or embedding is None:
            logger.warning("All loading attempts failed, creating a random embedding")
            embedding = torch.randn(1, 256)
            # Normalize it
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Check the shape and fix if needed
        if hasattr(embedding, 'shape'):
            logger.info(f"Loaded embedding with shape: {embedding.shape}")
            
            # Make sure it's the right shape - should be [1, 256]
            if embedding.shape != (1, 256):
                logger.warning(f"Unexpected tensor shape {embedding.shape}, reshaping to [1, 256]")
                # Reshape or pad as needed
                if embedding.numel() >= 256:
                    embedding = embedding.flatten()[:256].reshape(1, 256)
                else:
                    # If not enough values, pad with zeros
                    temp = torch.zeros(1, 256)
                    temp[0, :embedding.numel()] = embedding.flatten()
                    embedding = temp
        else:
            logger.warning("Loaded object has no shape attribute, creating new embedding")
            embedding = torch.randn(1, 256)
        
        # Normalize the embedding (common for speaker embeddings)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with current PyTorch version
        torch.save(embedding, output_path)
        logger.info(f"Successfully created new embedding file at {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to process embedding: {str(e)}")
        logger.warning("Creating a placeholder embedding instead")
        
        # Create a placeholder embedding as fallback
        placeholder = torch.zeros(1, 256)
        placeholder = torch.nn.functional.normalize(placeholder, p=2, dim=1)
        
        # Save the placeholder
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(placeholder, output_path)
        logger.info(f"Created placeholder embedding at {output_path}")

if __name__ == "__main__":
    main()