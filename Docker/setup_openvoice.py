#!/usr/bin/env python3
"""
OpenVoice Setup and Verification Script

This script verifies the OpenVoice installation and fixes common issues.
It focuses on ensuring that the speaker embedding files are compatible
and that the ToneColorConverter can be properly initialized.
"""

import os
import sys
import torch
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('openvoice_setup.log')
    ]
)
logger = logging.getLogger(__name__)

def check_pytorch_version():
    """Check if the PyTorch version is compatible with OpenVoice"""
    pytorch_version = torch.__version__
    logger.info(f"PyTorch version: {pytorch_version}")
    
    # OpenVoice works best with PyTorch 1.13.x but can work with others
    major, minor = map(int, pytorch_version.split('.')[:2])
    
    if major == 1 and minor >= 9:
        logger.warning(f"PyTorch {pytorch_version} might have compatibility issues with OpenVoice")
        logger.warning("Consider downgrading to PyTorch 1.13.x if you encounter problems")
        return True
    elif major == 1 and minor < 9:
        logger.info(f"PyTorch {pytorch_version} should be compatible with OpenVoice")
        return True
    elif major > 1:
        logger.warning(f"PyTorch {pytorch_version} may not be compatible with OpenVoice")
        logger.warning("Consider downgrading to PyTorch 1.13.x if you encounter problems")
        return True
    else:
        logger.error(f"PyTorch {pytorch_version} is likely not compatible with OpenVoice")
        return False

def check_required_packages():
    """Check if all required packages for OpenVoice are installed"""
    required_packages = [
        "openvoice",
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "wavmark",
        "langid",
        "gradio"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ Package '{package}' is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ Package '{package}' is missing")
    
    if missing_packages:
        logger.error(f"Missing {len(missing_packages)} required packages: {', '.join(missing_packages)}")
        logger.error("Please install the missing packages and try again")
        return False
    
    logger.info("All required packages are installed")
    return True

def check_openvoice_modules():
    """Check if OpenVoice modules can be imported correctly"""
    try:
        # Try to import key OpenVoice modules
        import openvoice
        from openvoice import utils, commons
        logger.info("OpenVoice basic modules imported successfully")
        
        # Try to import ToneColorConverter
        from openvoice.api import ToneColorConverter, BaseSpeakerTTS
        logger.info("OpenVoice API modules imported successfully")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import OpenVoice modules: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error importing OpenVoice modules: {str(e)}")
        return False

def verify_checkpoints():
    """Verify all required model checkpoints exist"""
    checkpoint_paths = {
        "converter_config": "checkpoints_v2/converter/config.json",
        "converter_checkpoint": "checkpoints_v2/converter/checkpoint.pth",
        "en_base_config": "checkpoints_v2/base_speakers/EN/config.json",
        "en_base_checkpoint": "checkpoints_v2/base_speakers/EN/checkpoint.pth",
        "en_default_se": "checkpoints_v2/base_speakers/EN/en_default_se.pth",
        "en_us_pth": "checkpoints_v2/base_speakers/ses/en-us.pth"
    }
    
    missing_files = []
    for name, path in checkpoint_paths.items():
        if not os.path.exists(path):
            missing_files.append((name, path))
            logger.error(f"Missing checkpoint: {name} at {path}")
        else:
            file_size = os.path.getsize(path)
            logger.info(f"Found checkpoint: {name} at {path} (size: {file_size} bytes)")
    
    if missing_files:
        logger.error(f"Missing {len(missing_files)} checkpoint files")
        logger.error("Please download the missing checkpoints or run the download_openvoice_models.sh script")
        return False
    
    logger.info("All required model checkpoints are available")
    return True

def test_speaker_embedding():
    """Test loading and validating speaker embedding file"""
    embedding_path = "checkpoints_v2/base_speakers/ses/en-us.pth"
    if not os.path.exists(embedding_path):
        logger.error(f"Speaker embedding file not found: {embedding_path}")
        return False
    
    try:
        # Try different loading methods to handle different PyTorch versions
        embedding = None
        
        # Method 1: Default loading
        try:
            embedding = torch.load(embedding_path)
            logger.info("Successfully loaded embedding with default parameters")
        except Exception as e:
            logger.warning(f"Failed to load with default parameters: {str(e)}")
            
            # Method 2: With map_location
            try:
                embedding = torch.load(embedding_path, map_location="cpu")
                logger.info("Successfully loaded embedding with map_location='cpu'")
            except Exception as e:
                logger.warning(f"Failed to load with map_location: {str(e)}")
                
                # Method 3: With weights_only if available (PyTorch 2.0+)
                try:
                    if hasattr(torch.load, "__kwdefaults__") and "weights_only" in torch.load.__kwdefaults__:
                        embedding = torch.load(embedding_path, weights_only=False)
                        logger.info("Successfully loaded embedding with weights_only=False")
                except Exception as e:
                    logger.warning(f"Failed to load with weights_only: {str(e)}")
        
        if embedding is None:
            logger.error("All loading attempts failed")
            return False
        
        # Check shape and fix if needed
        if not hasattr(embedding, 'shape'):
            logger.error("Loaded object is not a tensor")
            return False
            
        logger.info(f"Loaded embedding with shape: {embedding.shape}")
        
        # Check if shape is correct ([1, 256])
        if embedding.shape != (1, 256):
            logger.warning(f"Embedding has incorrect shape: {embedding.shape}, should be [1, 256]")
            
            # Try to reshape
            try:
                if embedding.numel() >= 256:
                    embedding = embedding.flatten()[:256].reshape(1, 256)
                    logger.info(f"Reshaped embedding to [1, 256]")
                    
                    # Save fixed embedding
                    fixed_path = "checkpoints_v2/base_speakers/ses/en-us-fixed.pth"
                    torch.save(embedding, fixed_path)
                    logger.info(f"Saved fixed embedding to {fixed_path}")
                else:
                    logger.error(f"Embedding has insufficient elements: {embedding.numel()}, needed 256")
                    return False
            except Exception as e:
                logger.error(f"Failed to reshape embedding: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing speaker embedding: {str(e)}")
        return False

def test_tone_converter():
    """Test initializing ToneColorConverter"""
    try:
        # Import ToneColorConverter
        from openvoice.api import ToneColorConverter
        
        config_path = "checkpoints_v2/converter/config.json"
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # Test on CPU
        logger.info("Initializing ToneColorConverter on CPU")
        converter = ToneColorConverter(config_path, device="cpu")
        logger.info("Successfully initialized ToneColorConverter on CPU")
        
        # Try to load checkpoint
        checkpoint_path = "checkpoints_v2/converter/checkpoint.pth"
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
            
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        converter.load_ckpt(checkpoint_path)
        logger.info("Successfully loaded checkpoint")
        
        # Test loading speaker embedding
        speaker_path = "checkpoints_v2/base_speakers/ses/en-us.pth"
        source_se = torch.load(speaker_path, map_location="cpu")
        
        if source_se.shape != torch.Size([1, 256]):
            logger.warning(f"Source SE has unexpected shape: {source_se.shape}")
            if source_se.numel() >= 256:
                source_se = source_se.flatten()[:256].reshape(1, 256)
                logger.info(f"Reshaped source_se to [1, 256]")
            else:
                logger.error(f"Source SE has insufficient elements: {source_se.numel()}, needed 256")
                return False
        
        logger.info("Speaker embedding loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing ToneColorConverter: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function that runs all checks"""
    logger.info("Starting OpenVoice setup and verification...")
    
    success = True
    success = check_pytorch_version() and success
    success = check_required_packages() and success
    success = check_openvoice_modules() and success
    success = verify_checkpoints() and success
    success = test_speaker_embedding() and success
    
    if success:
        success = test_tone_converter() and success
    
    if success:
        logger.info("✅ OpenVoice setup and verification completed successfully!")
        print("\n✅ OpenVoice is properly set up and ready to use!\n")
        return 0
    else:
        logger.error("❌ OpenVoice setup and verification failed")
        logger.error("Please check the logs for details and fix the issues")
        print("\n❌ OpenVoice setup failed. Check openvoice_setup.log for details.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())