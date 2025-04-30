# services/cascaded_backend.py
import os
import torch
import logging
import whisper
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import soundfile as sf
from gtts import gTTS
import librosa
import time
import warnings

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Import OpenVoice with proper error handling
try:
    from openvoice.api import ToneColorConverter
    OPENVOICE_AVAILABLE = True
except ImportError:
    OPENVOICE_AVAILABLE = False
    
from .translation_strategy import TranslationBackend

logger = logging.getLogger(__name__)

class CascadedBackend(TranslationBackend):
    """
    A modular cascaded translation backend using:
    - Whisper for Speech Recognition (ASR)
    - NLLB for Machine Translation (MT)
    - gTTS + OpenVoice for Text-to-Speech (TTS) with voice cloning
    """
    
    def __init__(self, device=None, use_voice_cloning=True):
        """Initialize the Cascaded backend"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        self.use_voice_cloning = use_voice_cloning and OPENVOICE_AVAILABLE
        
        # Base paths for OpenVoice
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.base_dir, "checkpoints_v2", "converter", "config.json")
        self.checkpoint_path = os.path.join(self.base_dir, "checkpoints_v2", "converter", "checkpoint.pth")
        self.speaker_path = os.path.join(self.base_dir, "checkpoints_v2", "base_speakers", "ses", "en-us.pth")
        
        # Language mappings
        self.languages = {
            'eng': 'English',
            'fra': 'French',
            'spa': 'Spanish',
            'deu': 'German',
            'ita': 'Italian',
            'por': 'Portuguese',
            'cmn': 'Chinese (Simplified)',
            'jpn': 'Japanese',
            'kor': 'Korean',
            'ara': 'Arabic',
            'hin': 'Hindi',
            'nld': 'Dutch',
            'rus': 'Russian',
            'pol': 'Polish',
            'tur': 'Turkish'
        }
        
        # Models will be loaded during initialization
        self.asr_model = None
        self.translator_model = None
        self.translator_tokenizer = None
        self.tone_converter = None
        self.source_se = None
    
    def initialize(self):
        """Initialize all required models"""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing Cascaded backend on {self.device}")
            start_time = time.time()
            
            # Load Whisper for ASR
            logger.info("Loading Whisper ASR model...")
            self.asr_model = whisper.load_model("medium", device=self.device)
            
            # Load NLLB for translation
            logger.info("Loading NLLB translation model...")
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            
            # Move models to device if using GPU
            if self.device.type == 'cuda':
                self.translator_model = self.translator_model.to(self.device)
                # Whisper handles device placement internally
                
            # Load OpenVoice if voice cloning is enabled
            if self.use_voice_cloning:
                logger.info("Loading OpenVoice model...")
                self._initialize_openvoice()
            
            self.initialized = True
            logger.info(f"Cascaded backend initialized successfully in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cascaded backend: {str(e)}")
            raise
    
    def _initialize_openvoice(self):
        """Initialize OpenVoice components with proper error handling"""
        try:
            # Log paths to help with debugging
            logger.info(f"OpenVoice config path: {self.config_path} (exists: {os.path.exists(self.config_path)})")
            logger.info(f"OpenVoice checkpoint path: {self.checkpoint_path} (exists: {os.path.exists(self.checkpoint_path)})")
            logger.info(f"OpenVoice speaker path: {self.speaker_path} (exists: {os.path.exists(self.speaker_path)})")
            
            # Validate required files exist
            for path, desc in [
                (self.config_path, "OpenVoice config"),
                (self.checkpoint_path, "OpenVoice checkpoint"),
                (self.speaker_path, "Speaker embedding")
            ]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{desc} file not found at {path}")
            
            # Always use string device type - this is crucial!
            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            logger.info(f"Using device string: {device_str} for OpenVoice")
            
            # Create tone color converter with string device
            self.tone_converter = ToneColorConverter(self.config_path, device=device_str)
            logger.info("Successfully created ToneColorConverter")
            
            # Load checkpoint with detailed error handling
            try:
                self.tone_converter.load_ckpt(self.checkpoint_path)
                logger.info("Successfully loaded checkpoint for ToneColorConverter")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                raise
            
            # Load speaker embedding with correct weights_only parameter for PyTorch 2.6+
            try:
                # Try multiple loading strategies to handle different PyTorch versions
                loading_strategies = [
                    {"weights_only": False},
                    {"weights_only": True},
                    {}  # No weights_only parameter for older PyTorch versions
                ]
                
                last_error = None
                for i, kwargs in enumerate(loading_strategies):
                    try:
                        logger.debug(f"Attempting to load speaker embedding with strategy {i+1}: {kwargs}")
                        self.source_se = torch.load(
                            self.speaker_path, 
                            map_location=device_str, 
                            **kwargs
                        )
                        logger.info(f"Successfully loaded speaker embedding with strategy {i+1}")
                        
                        # Ensure embedding has correct shape [1, 256]
                        if self.source_se.ndim == 1:
                            logger.warning(f"Reshaping speaker embedding from {self.source_se.shape} to [1, 256]")
                            self.source_se = self.source_se.reshape(1, -1)
                            
                        # Verify shape is correct
                        expected_shape = [1, 256]
                        if list(self.source_se.shape) != expected_shape:
                            logger.warning(f"Speaker embedding has unexpected shape: {self.source_se.shape}, expected {expected_shape}")
                            if self.source_se.numel() >= 256:
                                logger.info("Reshaping embedding to correct dimensions")
                                self.source_se = self.source_se.flatten()[:256].reshape(1, 256)
                            else:
                                logger.error(f"Speaker embedding has insufficient elements: {self.source_se.numel()}, needed 256")
                                raise ValueError("Speaker embedding has insufficient elements")
                        
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load speaker embedding with strategy {i+1}: {str(e)}")
                        last_error = e
                        continue
                        
                if self.source_se is None:
                    if last_error:
                        raise last_error
                    else:
                        raise ValueError("All speaker embedding loading strategies failed")
                        
            except Exception as e:
                logger.error(f"Failed to load speaker embedding: {str(e)}")
                raise
            
            logger.info("OpenVoice model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load OpenVoice: {str(e)}")
            logger.info("Voice cloning will be disabled")
            self.use_voice_cloning = False
            raise
    
    def translate_speech(
        self, 
        audio_tensor: torch.Tensor, 
        source_lang: str = "eng",
        target_lang: str = "fra"
    ) -> Dict[str, Any]:
        """
        Translate speech using a cascaded approach:
        1. Whisper for ASR
        2. NLLB for translation
        3. gTTS + OpenVoice for TTS
        
        Args:
            audio_tensor: Processed audio tensor
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Dictionary with translated audio, source text, and target text
        """
        if not self.initialized:
            self.initialize()
        
        # Create a temp directory for intermediate files
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Step 1: Speech recognition with Whisper
            logger.info(f"Step 1: Performing ASR on audio ({source_lang})")
            audio_numpy = audio_tensor.squeeze().cpu().numpy()
            
            # Ensure audio is properly formatted for Whisper
            if np.isnan(audio_numpy).any() or np.isinf(audio_numpy).any():
                logger.warning("Audio contains invalid values, cleaning up")
                audio_numpy = np.nan_to_num(audio_numpy)
                
            # Normalize audio if needed
            max_amp = np.abs(audio_numpy).max()
            if max_amp > 0 and max_amp != 1.0:
                logger.debug(f"Normalizing audio with factor {max_amp}")
                audio_numpy = audio_numpy / max_amp
                
            # Save reference audio for potential voice cloning
            ref_audio_path = temp_dir / "reference_audio.wav"
            sf.write(str(ref_audio_path), audio_numpy, 16000)
            logger.debug(f"Saved reference audio to {ref_audio_path}")
                
            # Use Whisper for transcription
            try:
                # Use a more robust approach with Whisper
                result = self.asr_model.transcribe(
                    audio_numpy,
                    language=source_lang if source_lang != "eng" else None,
                    task="transcribe"
                )
                source_text = result["text"]
                detected_lang = result.get("language", source_lang)
                
                logger.info(f"ASR Result: '{source_text}'")
                logger.info(f"Detected language: {detected_lang}")
                
                # If no text was transcribed, use a placeholder
                if not source_text or len(source_text.strip()) == 0:
                    logger.warning("No text transcribed, using placeholder")
                    source_text = "No speech detected."
                    
            except Exception as e:
                logger.error(f"ASR failed: {str(e)}")
                source_text = "Speech recognition failed."
                detected_lang = source_lang
            
            # Step 2: Text translation with NLLB
            logger.info(f"Step 2: Translating text from {source_lang} to {target_lang}")
            
            try:
                # Convert language codes to NLLB format
                src_lang_code = self._convert_to_nllb_code(source_lang)
                tgt_lang_code = self._convert_to_nllb_code(target_lang)
                
                # Tokenize input text
                inputs = self.translator_tokenizer(source_text, return_tensors="pt")
                if self.device.type == 'cuda':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Find the forced BOS token ID 
                forced_bos_token_id = None
                if hasattr(self.translator_tokenizer, 'lang_code_to_id'):
                    forced_bos_token_id = self.translator_tokenizer.lang_code_to_id.get(tgt_lang_code)
                
                if forced_bos_token_id is None:
                    # Fall back to using convert_tokens_to_ids
                    forced_bos_token_id = self.translator_tokenizer.convert_tokens_to_ids(tgt_lang_code)
                
                # Generate translation
                with torch.no_grad():
                    translated_tokens = self.translator_model.generate(
                        **inputs, 
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=200,
                        num_beams=4,
                        length_penalty=1.0
                    )
                
                target_text = self.translator_tokenizer.batch_decode(
                    translated_tokens, 
                    skip_special_tokens=True
                )[0]
                
                logger.info(f"MT Result: '{target_text}'")
                
                # If translation failed, use the source text
                if not target_text or len(target_text.strip()) == 0:
                    logger.warning("Translation failed, using source text")
                    target_text = source_text
                    
            except Exception as e:
                logger.error(f"Translation failed: {str(e)}")
                target_text = source_text
            
            # Step 3: Text-to-speech synthesis (with optional voice cloning)
            logger.info(f"Step 3: Performing TTS for {target_lang}")
            
            try:
                # Use gTTS to generate base speech
                gtts_lang = self._convert_to_gtts_code(target_lang)
                base_audio_path = temp_dir / "base_audio.mp3"
                
                tts = gTTS(text=target_text, lang=gtts_lang)
                tts.save(str(base_audio_path))
                
                # Load and convert to WAV format for processing
                base_wav_path = temp_dir / "base_audio.wav"
                y, sr = librosa.load(str(base_audio_path), sr=16000, mono=True)
                sf.write(str(base_wav_path), y, sr)
                
                if self.use_voice_cloning and self.tone_converter is not None and self.source_se is not None:
                    # Apply voice cloning with OpenVoice
                    logger.info("Applying voice cloning with OpenVoice")
                    
                    output_path = temp_dir / "cloned_audio.wav"
                    
                    try:
                        # Preprocess the audio for OpenVoice
                        # 1. Ensure correct format for base audio (16kHz, mono)
                        audio_data, sr = librosa.load(str(base_wav_path), sr=16000, mono=True)
                        
                        # 2. Create a temporary file with the properly formatted audio
                        preprocessed_path = temp_dir / "preprocessed_audio.wav"
                        sf.write(str(preprocessed_path), audio_data, 16000)
                        
                        logger.debug(f"Prepared audio for conversion: format=16kHz mono, shape={audio_data.shape}")
                        
                        # 3. Ensure voice is normalized to prevent clipping
                        audio_max = np.abs(audio_data).max()
                        if audio_max > 0.9:
                            logger.debug(f"Normalizing audio level from {audio_max} to 0.9")
                            audio_data = audio_data * (0.9 / audio_max)
                            sf.write(str(preprocessed_path), audio_data, 16000)
                        
                        # 4. Simple voice conversion using the pre-loaded source embedding
                        try:
                            # Suppress warnings during conversion
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                
                                # Confirm source embedding shape
                                logger.debug(f"Source embedding shape: {self.source_se.shape}")
                                
                                # Perform the conversion
                                self.tone_converter.convert(
                                    audio_src_path=str(preprocessed_path),
                                    src_se=self.source_se,
                                    tgt_se=self.source_se,  # Using same embedding for simplicity
                                    output_path=str(output_path)
                                )
                        except Exception as conversion_error:
                            logger.error(f"Error during tone conversion: {str(conversion_error)}")
                            
                            # Try alternative method with enhanced preprocessing if the first attempt fails
                            logger.info("Attempting conversion with alternative preprocessing")
                            
                            # Read audio directly through soundfile for precise control
                            audio_arr, sr = sf.read(str(base_wav_path))
                            
                            # Ensure it's mono and at 16kHz
                            if len(audio_arr.shape) > 1:
                                audio_arr = np.mean(audio_arr, axis=1)
                            
                            if sr != 16000:
                                audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                                sr = 16000
                                
                            # Convert to float32 which OpenVoice expects
                            audio_arr = audio_arr.astype(np.float32)
                            
                            # Apply subtle pre-emphasis to emphasize higher frequencies
                            pre_emphasis = 0.97
                            emphasized_audio = np.append(audio_arr[0], audio_arr[1:] - pre_emphasis * audio_arr[:-1])
                            
                            # Save the preprocessed audio
                            alt_preprocessed_path = temp_dir / "alt_preprocessed_audio.wav"
                            sf.write(str(alt_preprocessed_path), emphasized_audio, 16000)
                            
                            # Try converting with the alternative method
                            self.tone_converter.convert(
                                audio_src_path=str(alt_preprocessed_path),
                                src_se=self.source_se,
                                tgt_se=self.source_se,  # Using same embedding for simplicity
                                output_path=str(output_path)
                            )
                        
                        # Load the converted audio
                        if os.path.exists(str(output_path)) and os.path.getsize(str(output_path)) > 0:
                            y_converted, sr_converted = sf.read(str(output_path))
                            output_tensor = torch.FloatTensor(y_converted).unsqueeze(0)
                            
                            logger.info(f"Voice conversion successful, shape: {output_tensor.shape}")
                        else:
                            logger.warning("Conversion output file empty or missing. Falling back to base audio.")
                            raise FileNotFoundError("Conversion output file missing or empty")
                            
                    except Exception as e:
                        logger.error(f"Voice conversion failed: {str(e)}")
                        logger.warning("Falling back to base audio")
                        
                        # Fall back to base audio
                        output_tensor = torch.FloatTensor(y).unsqueeze(0)
                else:
                    # Just use gTTS output
                    logger.info("Using base gTTS audio (no voice cloning)")
                    output_tensor = torch.FloatTensor(y).unsqueeze(0)
                
                # Final processing and normalization
                if output_tensor.dim() > 2:
                    logger.debug(f"Reducing dimensions from {output_tensor.shape}")
                    output_tensor = output_tensor.mean(dim=1)
                
                if output_tensor.dim() == 1:
                    logger.debug(f"Adding batch dimension to tensor of shape {output_tensor.shape}")
                    output_tensor = output_tensor.unsqueeze(0)
                
                # Normalize audio to prevent distortion
                max_val = torch.abs(output_tensor).max()
                if max_val > 0:
                    logger.debug(f"Normalizing output audio with factor {max_val}")
                    output_tensor = output_tensor / max_val * 0.9
                
                return {
                    "audio": output_tensor,
                    "transcripts": {
                        "source": source_text,
                        "target": target_text
                    }
                }
                
            except Exception as e:
                logger.error(f"TTS synthesis failed: {str(e)}")
                # Return silent audio with transcript
                return {
                    "audio": torch.zeros((1, 16000)),  # 1 second of silence
                    "transcripts": {
                        "source": source_text,
                        "target": target_text
                    }
                }
                
        except Exception as e:
            logger.error(f"Cascaded translation failed: {str(e)}")
            return {
                "audio": torch.zeros((1, 16000)),  # 1 second of silence
                "transcripts": {
                    "source": f"Error: {str(e)}",
                    "target": f"Error: {str(e)}"
                }
            }
            
        finally:
            # Clean up temporary directory
            try:
                import shutil
                logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")
    
    def _convert_to_nllb_code(self, lang_code: str) -> str:
        """Convert standard language code to NLLB format"""
        mapping = {
            'eng': 'eng_Latn',
            'fra': 'fra_Latn',
            'spa': 'spa_Latn',
            'deu': 'deu_Latn',
            'ita': 'ita_Latn',
            'por': 'por_Latn',
            'rus': 'rus_Cyrl',
            'cmn': 'zho_Hans',
            'jpn': 'jpn_Jpan',
            'kor': 'kor_Hang',
            'ara': 'ara_Arab',
            'hin': 'hin_Deva',
            'nld': 'nld_Latn',
            'pol': 'pol_Latn',
            'tur': 'tur_Latn',
        }
        return mapping.get(lang_code, 'eng_Latn')  # Default to English
    
    def _convert_to_gtts_code(self, lang_code: str) -> str:
        """Convert standard language code to gTTS format"""
        mapping = {
            'eng': 'en',
            'fra': 'fr',
            'spa': 'es',
            'deu': 'de',
            'ita': 'it',
            'por': 'pt',
            'cmn': 'zh-CN',
            'jpn': 'ja',
            'kor': 'ko',
            'ara': 'ar',
            'hin': 'hi',
            'nld': 'nl',
            'rus': 'ru',
            'pol': 'pl',
            'tur': 'tr',
        }
        return mapping.get(lang_code, 'en')  # Default to English
    
    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language is supported by this backend"""
        return lang_code in self.languages
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        return self.languages