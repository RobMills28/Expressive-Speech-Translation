# test_espnet_backend.py

import logging
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_espnet_backend")

# Import the ESPnet backend
from services.espnet_backend import ESPnetBackend

def test_asr_component():
    """Test the ASR component of the backend"""
    try:
        logger.info("Testing ASR component...")
        backend = ESPnetBackend()
        result = backend._load_asr_model('eng')
        logger.info(f"ASR model loaded: {result}")
        return result
    except Exception as e:
        logger.error(f"ASR component test failed: {str(e)}")
        return False

def test_tts_component():
    """Test the TTS component of the backend"""
    try:
        logger.info("Testing TTS component...")
        backend = ESPnetBackend()
        
        # Test English TTS
        eng_result = backend._load_tts_model('eng')
        logger.info(f"English TTS model loaded: {eng_result}")
        
        return eng_result
    except Exception as e:
        logger.error(f"TTS component test failed: {str(e)}")
        return False

def test_end_to_end(audio_path):
    """Test the full end-to-end pipeline with improved error handling"""
    try:
        logger.info(f"Testing end-to-end with audio: {audio_path}")
        
        # Load audio file
        audio, rate = sf.read(audio_path)
        logger.info(f"Loaded audio: shape={audio.shape}, rate={rate}Hz")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio[:, 0]
            logger.info(f"Converted to mono: shape={audio.shape}")
        
        # Resample to 16kHz if needed
        if rate != 16000:
            logger.warning(f"Audio sample rate is {rate}Hz, ESPnet expects 16000Hz")
            # Better resampling using scipy
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / rate))
            logger.info(f"Resampled audio: shape={audio.shape}")
        
        # Ensure audio has values and is normalized
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.9
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        
        # Initialize backend
        backend = ESPnetBackend()
        
        # Perform English to English translation (simplest test case)
        logger.info("Performing English to English translation...")
        result = backend.translate_speech(
            audio_tensor=audio_tensor,
            source_lang="eng",
            target_lang="eng"  # Use English to English for initial testing
        )
        
        # Verify result structure
        if not isinstance(result, dict):
            logger.error(f"Result is not a dictionary: {type(result)}")
            return False
            
        if "audio" not in result or "transcripts" not in result:
            logger.error(f"Result missing required keys: {result.keys()}")
            return False
            
        # Log results
        logger.info(f"Source text: {result['transcripts']['source']}")
        logger.info(f"Translated text: {result['transcripts']['target']}")
        logger.info(f"Generated audio shape: {result['audio'].shape}")
        
        # Save output audio
        output_path = Path(audio_path).parent / "espnet_output.wav"
        sf.write(output_path, result['audio'].squeeze().numpy(), 16000)
        logger.info(f"Saved output audio to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"End-to-end test failed: {str(e)}")
        return False

def main():
    logger.info("ESPnet Backend Test")
    logger.info("===================")
    
    if len(sys.argv) < 2:
        logger.error("Please provide an audio file path")
        logger.info("Usage: python test_espnet_backend.py /path/to/audio.wav")
        return
    
    audio_path = sys.argv[1]
    
    # Test components individually
    logger.info("\n=== Component Tests ===")
    asr_success = test_asr_component()
    tts_success = test_tts_component()
    
    if asr_success and tts_success:
        logger.info("\nAll components loaded successfully!")
        
        # Test full pipeline
        logger.info("\n=== End-to-End Test ===")
        e2e_success = test_end_to_end(audio_path)
        
        if e2e_success:
            logger.info("\n✅ All tests passed!")
        else:
            logger.error("\n❌ End-to-end test failed")
    else:
        logger.error("\n❌ Component tests failed")

if __name__ == "__main__":
    main()