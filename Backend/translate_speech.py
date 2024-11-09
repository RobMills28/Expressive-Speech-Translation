from transformers import SeamlessM4TProcessor, SeamlessM4TModel
import torch
import torchaudio
import os
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000  # SeamlessM4T expects 16kHz
MODEL_NAME = "facebook/seamless-m4t-v2-large"

# Load the processor and model with token
try:
    processor = SeamlessM4TProcessor.from_pretrained(
        MODEL_NAME,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    model = SeamlessM4TModel.from_pretrained(
        MODEL_NAME,
        token=os.getenv('HUGGINGFACE_TOKEN')
    )
    logger.info("Model and processor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def process_audio(input_path):
    """
    Process audio file to match model requirements
    """
    try:
        logger.info(f"Processing audio file: {input_path}")
        
        # Load audio file
        audio, orig_freq = torchaudio.load(input_path)
        logger.info(f"Original audio shape: {audio.shape}, frequency: {orig_freq}Hz")
        
        # Resample to 16kHz
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=SAMPLE_RATE)
        logger.info(f"Resampled audio shape: {audio.shape}")
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            logger.info("Converted stereo to mono")
        
        # Normalize audio
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
            logger.info("Normalized audio")
            
        return audio
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def translate_audio(input_path, target_language="fra"):
    """
    Process and translate audio file
    
    Args:
        input_path (str): Path to input audio file
        target_language (str): Target language code (default: "fra")
    
    Returns:
        tuple: (waveform_tensor, sample_rate)
    """
    try:
        logger.info(f"Starting translation to {target_language}")
        
        # Process audio
        audio = process_audio(input_path)
        audio_numpy = audio.squeeze().numpy()
        
        # Prepare model inputs
        logger.info("Preparing model inputs")
        inputs = processor(
            audios=audio_numpy,
            sampling_rate=SAMPLE_RATE,
            src_lang="eng",  # Source language is English
            tgt_lang=target_language,
            return_tensors="pt"
        )
        
        # Generate translation
        logger.info("Generating translation")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                tgt_lang=target_language,
                num_beams=5,
                max_new_tokens=200,
                use_cache=True
            )
        
        # Process the output
        logger.info("Processing model outputs")
        if hasattr(outputs, 'waveform'):
            audio_output = outputs.waveform[0].cpu().numpy()
            logger.info("Used waveform attribute")
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            audio_output = outputs[0].cpu().numpy()
            logger.info("Used tuple format")
        else:
            raise ValueError("Unexpected output format from model")
        
        # Convert to torch tensor
        waveform_tensor = torch.tensor(audio_output).unsqueeze(0)
        
        # Normalize output audio
        if waveform_tensor.abs().max() > 1.0:
            waveform_tensor = waveform_tensor / waveform_tensor.abs().max()
            logger.info("Normalized output audio")
        
        logger.info(f"Translation complete. Output shape: {waveform_tensor.shape}")
        return waveform_tensor, SAMPLE_RATE
        
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise

def save_translation(waveform_tensor, input_path, output_dir=None):
    """
    Save the translated audio to a file
    """
    try:
        if output_dir is None:
            output_dir = os.path.dirname(input_path)
        
        output_filename = f"translated_{Path(input_path).stem}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        torchaudio.save(output_path, waveform_tensor, sample_rate=SAMPLE_RATE)
        logger.info(f"Saved translation to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving translation: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage / testing
    try:
        test_path = r"/Users/robmills/Documents/Audio Samples/English (US)/Arthur.mp3"
        waveform, sr = translate_audio(test_path, "fra")
        output_path = save_translation(waveform, test_path)
        print(f"Translation complete. Output saved as {output_path}")
    except Exception as e:
        print(f"Test failed: {str(e)}")