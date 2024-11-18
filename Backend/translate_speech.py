from transformers import SeamlessM4TProcessor, SeamlessM4TModel
import torch
import torchaudio
import os
import numpy as np
from pathlib import Path
import logging
import scipy.signal

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
        
        # Resample to 16kHz with high-quality settings
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq,
            new_freq=SAMPLE_RATE,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492
        )
        audio = resampler(audio)
        logger.info(f"Resampled audio shape: {audio.shape}")
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            logger.info("Converted stereo to mono")
        
        # Remove DC offset
        audio = audio - torch.mean(audio, dim=1, keepdim=True)
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio = torch.cat([audio[:, :1], audio[:, 1:] - pre_emphasis * audio[:, :-1]], dim=1)
        
        # Normalize with headroom
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio * (0.95 / max_val)
            
        return audio
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def translate_audio(input_path, target_language="fra"):
    """
    Process and translate audio file with improved audio quality
    """
    try:
        logger.info(f"Starting translation to {target_language}")
        
        # Enhanced audio processing
        audio = process_audio(input_path)
        
        # Apply bandpass filter for speech frequencies
        audio_numpy = audio.squeeze().numpy()
        nyquist = SAMPLE_RATE // 2
        low_cutoff = 80 / nyquist
        high_cutoff = 7500 / nyquist
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_audio = scipy.signal.filtfilt(b, a, audio_numpy)
        
        # Normalize filtered audio
        filtered_audio = filtered_audio / np.max(np.abs(filtered_audio)) * 0.95
        filtered_audio = filtered_audio - np.mean(filtered_audio)
        
        # Prepare model inputs
        logger.info("Preparing model inputs")
        inputs = processor(
            audios=filtered_audio,
            sampling_rate=SAMPLE_RATE,
            src_lang="eng",
            tgt_lang=target_language,
            return_tensors="pt"
        )
        
        # Generate translation with improved parameters
        logger.info("Generating translation")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                tgt_lang=target_language,
                num_beams=5,
                max_new_tokens=200,
                use_cache=True,
                temperature=0.7,
                length_penalty=1.0
            )
        
        # Process the output with enhanced audio quality
        logger.info("Processing model outputs")
        if hasattr(outputs, 'waveform'):
            audio_output = outputs.waveform[0].cpu().numpy()
            logger.info("Used waveform attribute")
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            audio_output = outputs[0].cpu().numpy()
            logger.info("Used tuple format")
        else:
            raise ValueError("Unexpected output format from model")
        
        # Post-process the output audio
        audio_output = audio_output - np.mean(audio_output)  # Remove DC offset
        audio_output = np.tanh(audio_output)  # Gentle limiting
        
        # Final normalization with headroom
        max_val = np.abs(audio_output).max()
        if max_val > 0:
            audio_output = audio_output * (0.95 / max_val)
        
        # Convert to torch tensor
        waveform_tensor = torch.tensor(audio_output).unsqueeze(0)
        
        logger.info(f"Translation complete. Output shape: {waveform_tensor.shape}")
        logger.info(f"Output range: {waveform_tensor.min():.3f} to {waveform_tensor.max():.3f}")
        
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