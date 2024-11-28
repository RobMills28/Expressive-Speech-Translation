import os
import torch
import torchaudio
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio file processing and validation for the translation service.
    """
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.ogg', '.flac'}
    MAX_AUDIO_LENGTH = 300  # seconds (5 minutes)
    SAMPLE_RATE = 16000

    @staticmethod
    def validate_audio_length(audio_path: str) -> tuple[bool, str]:
        """
        Validates audio file length and basic integrity
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Format validation
            if Path(audio_path).suffix.lower() not in AudioProcessor.SUPPORTED_FORMATS:
                return False, f"Unsupported audio format. Supported: {AudioProcessor.SUPPORTED_FORMATS}"

            # File checks
            if not os.path.exists(audio_path):
                return False, "Audio file not found"
                
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                return False, "Audio file is empty"

            # Get audio metadata
            metadata = torchaudio.info(audio_path)
            
            # Check sample rate
            if metadata.sample_rate <= 0:
                return False, "Invalid sample rate detected"
                
            # Check number of frames
            if metadata.num_frames <= 0:
                return False, "No audio frames detected"

            # Calculate duration
            duration = metadata.num_frames / metadata.sample_rate
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            if duration <= 0:
                return False, "Invalid audio duration"
                
            if duration > AudioProcessor.MAX_AUDIO_LENGTH:
                return False, f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({AudioProcessor.MAX_AUDIO_LENGTH}s)"

            return True, ""

        except Exception as e:
            error_msg = f"Error validating audio: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    def process_audio(audio_path: str) -> torch.Tensor:
        """
        Processes an audio file for translation.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            torch.Tensor: Processed audio tensor ready for model input
            
        Raises:
            ValueError: If audio processing fails
        """
        try:
            logger.info(f"Loading audio from: {audio_path}")
            
            # Load and check audio
            info = torchaudio.info(audio_path)
            logger.info(f"Audio info - Sample rate: {info.sample_rate}, Channels: {info.num_channels}")
            
            audio, orig_freq = torchaudio.load(audio_path)
            
            # Quality checks
            if torch.isnan(audio).any():
                raise ValueError("Audio contains NaN values")
            if torch.isinf(audio).any():
                raise ValueError("Audio contains infinite values")
            if audio.abs().max() == 0:
                raise ValueError("Audio is silent")
            
            logger.info(f"Original audio shape: {audio.shape}, Frequency: {orig_freq}Hz")
            
            # Process in chunks if large
            if audio.shape[1] > 1_000_000:  # If longer than 1M samples
                chunks = audio.split(1_000_000, dim=1)
                audio = torch.cat([chunk for chunk in chunks], dim=1)
            
            # Resample if needed
            if orig_freq != AudioProcessor.SAMPLE_RATE:
                logger.info(f"Resampling from {orig_freq}Hz to {AudioProcessor.SAMPLE_RATE}Hz")
                audio = torchaudio.functional.resample(
                    audio, 
                    orig_freq=orig_freq, 
                    new_freq=AudioProcessor.SAMPLE_RATE
                )
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                logger.info("Converting from stereo to mono")
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Normalize audio
            if audio.abs().max() > 1.0:
                logger.info("Normalizing audio")
                audio = audio / audio.abs().max()
            
            logger.info(f"Processed audio shape: {audio.shape}")
            return audio
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process audio: {str(e)}")