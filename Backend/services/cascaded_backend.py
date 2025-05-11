# services/cascaded_backend.py
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import soundfile as sf
from gtts import gTTS
import librosa
import time
import warnings
from pydub import AudioSegment

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Try to import whisper, but make it optional
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("Whisper not available - will use transformers pipeline instead")
    WHISPER_AVAILABLE = False

# Use OpenVoice Docker API instead of direct imports
import requests

# Check if Docker API is available
def check_openvoice_api():
    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True
        return False
    except Exception as e:
        print(f"OpenVoice API not available: {e}")
        return False

OPENVOICE_AVAILABLE = check_openvoice_api()
    
from .translation_strategy import TranslationBackend

logger = logging.getLogger(__name__)

class CascadedBackend(TranslationBackend):
    """
    A modular cascaded translation backend using:
    - Whisper or transformers pipeline for Speech Recognition (ASR)
    - NLLB for Machine Translation (MT)
    - gTTS + OpenVoice for Text-to-Speech (TTS) with voice cloning
    """
    
    def __init__(self, device=None, use_voice_cloning=True):
        """Initialize the Cascaded backend"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        self.use_voice_cloning = use_voice_cloning and OPENVOICE_AVAILABLE
        
        # Base paths for OpenVoice API (not direct integration anymore)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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
        self.asr_pipeline = None
        self.translator_model = None
        self.translator_tokenizer = None
    
    def initialize(self):
        """Initialize all required models"""
        if self.initialized:
            return
            
        try:
            logger.info(f"Initializing Cascaded backend on {self.device}")
            start_time = time.time()
            
            # Load ASR model - either whisper or transformers pipeline
            logger.info("Loading ASR model...")
            if WHISPER_AVAILABLE:
                self.asr_model = whisper.load_model("medium", device=self.device)
            else:
                # Use transformers pipeline as an alternative
                logger.info("Using transformers pipeline for ASR")
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-small",
                    device=0 if self.device.type == 'cuda' else -1
                )
            
            # Load NLLB for translation
            logger.info("Loading NLLB translation model...")
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            
            # Move models to device if using GPU
            if self.device.type == 'cuda':
                self.translator_model = self.translator_model.to(self.device)
                # Whisper handles device placement internally
            
            self.initialized = True
            logger.info(f"Cascaded backend initialized successfully in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cascaded backend: {str(e)}")
            raise
    
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
    
    def _clone_voice_with_api(self, input_audio_path, target_text, output_audio_path):
        """Clone voice using the OpenVoice Docker API with text input"""
        if not OPENVOICE_AVAILABLE:
            logger.warning("OpenVoice API is not available")
            return False
            
        try:
            logger.info(f"Attempting voice cloning with OpenVoice API")
            logger.info(f"Text to be synthesized: {target_text[:100]}...")
            
            # Step 1: Generate base audio with gTTS (text-to-speech)
            temp_dir = os.path.dirname(output_audio_path)
            base_audio_path = os.path.join(temp_dir, "base_tts.wav")
            
            # Generate TTS audio from the translated text
            tts = gTTS(text=target_text, lang='en', slow=False)
            temp_mp3 = os.path.join(temp_dir, "temp.mp3")
            tts.save(temp_mp3)
            
            # Convert MP3 to WAV
            sound = AudioSegment.from_mp3(temp_mp3)
            sound.export(base_audio_path, format="wav")
            logger.info(f"Generated base TTS audio at {base_audio_path} ({os.path.getsize(base_audio_path)} bytes)")
            
            # Step 2: Send both files to OpenVoice API for voice cloning
            logger.info(f"Sending files to OpenVoice API:")
            logger.info(f"- Reference voice: {input_audio_path} ({os.path.getsize(input_audio_path)} bytes)")
            logger.info(f"- Base TTS: {base_audio_path} ({os.path.getsize(base_audio_path)} bytes)")
            
            # Send both files to the API
            with open(input_audio_path, "rb") as f_source, open(base_audio_path, "rb") as f_target:
                files = {
                    "audio_file": (os.path.basename(input_audio_path), f_source, "audio/wav"),
                    "target_file": (os.path.basename(base_audio_path), f_target, "audio/wav")
                }
                response = requests.post("http://localhost:8000/clone-voice", files=files, timeout=30)
            
            # Check response
            if response.status_code == 200:
                # Save response to output file
                with open(output_audio_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Voice cloning successful via API: {os.path.getsize(output_audio_path)} bytes")
                
                # Clean up temporary files
                try:
                    os.remove(temp_mp3)
                    os.remove(base_audio_path)
                except:
                    pass
                    
                return True
            else:
                logger.error(f"Voice cloning API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error during API voice cloning: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
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
        3. gTTS + OpenVoice for voice cloning
        """
        if not self.initialized:
            self.initialize()
        
        # Start timer for performance tracking
        start_time = time.time()
        
        # Create temp dir for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Source and target language names for gTTS
            src_name = self.languages.get(source_lang, "English")
            tgt_name = self.languages.get(target_lang, "French")
            
            try:
                # 1. Speech recognition (ASR) using Whisper or transformers pipeline
                logger.info(f"Starting ASR for {src_name}")
                
                # Use whisper or transformers pipeline to transcribe the audio
                audio_numpy = audio_tensor.squeeze().numpy()
                
                # Normalize audio
                if np.abs(audio_numpy).max() > 0:
                    audio_numpy = audio_numpy / np.abs(audio_numpy).max()
                
                # Process with either whisper or transformers pipeline
                if WHISPER_AVAILABLE and self.asr_model is not None:
                    # Use whisper directly
                    asr_result = self.asr_model.transcribe(
                        audio_numpy,
                        language=self._convert_to_gtts_code(source_lang),
                        task="transcribe",
                        fp16=False
                    )
                    source_text = asr_result["text"]
                else:
                    # Use transformers pipeline
                    # Save audio to a temporary file
                    temp_audio_path = temp_dir / "temp_audio.wav"
                    sf.write(str(temp_audio_path), audio_numpy, 16000)
                    
                    # Run ASR with transformers
                    asr_result = self.asr_pipeline(str(temp_audio_path))
                    source_text = asr_result["text"] if isinstance(asr_result, dict) else asr_result
                
                logger.info(f"ASR result: {source_text}")
                
                # 2. Text translation with NLLB model
                logger.info(f"Translating text from {src_name} to {tgt_name}")
                
                # Prepare for NLLB translation
                src_code = self._convert_to_nllb_code(source_lang)
                tgt_code = self._convert_to_nllb_code(target_lang)
                
                # Encode with tokenizer
                input_ids = self.translator_tokenizer(
                    source_text, 
                    return_tensors="pt", 
                    padding=True
                ).input_ids.to(self.device)
                
                # Generate translation
                with torch.no_grad():
                    translated_tokens = self.translator_model.generate(
                        input_ids,
                        forced_bos_token_id=self.translator_tokenizer.lang_code_to_id[tgt_code],
                        max_length=1024,
                        num_beams=5,
                        length_penalty=1.0
                    )
                
                # Decode back to text
                target_text = self.translator_tokenizer.batch_decode(
                    translated_tokens, 
                    skip_special_tokens=True
                )[0]
                
                logger.info(f"Translated text: {target_text}")
                
                # 3. Text-to-Speech (TTS) with gTTS and voice cloning
                logger.info(f"Generating speech in {tgt_name} with gTTS")
                
                # Base audio file path
                base_wav_path = temp_dir / "base_audio.wav"
                
                # Use gTTS to generate base audio
                tts = gTTS(
                    text=target_text,
                    lang=self._convert_to_gtts_code(target_lang),
                    slow=False
                )
                
                # Save as MP3 first
                temp_mp3 = temp_dir / "temp.mp3"
                tts.save(str(temp_mp3))
                
                # Convert MP3 to WAV for further processing
                sound = AudioSegment.from_mp3(str(temp_mp3))
                sound.export(str(base_wav_path), format="wav")
                
                # Check if base audio file exists and has content
                if not base_wav_path.exists() or base_wav_path.stat().st_size == 0:
                    raise ValueError("Failed to generate base audio with gTTS")
                
                # Voice cloning with OpenVoice if available
                if self.use_voice_cloning and OPENVOICE_AVAILABLE:
                    # Apply voice cloning with OpenVoice API
                    logger.info("Applying voice cloning with OpenVoice API")
                    
                    output_path = temp_dir / "cloned_audio.wav"
                    
                    try:
                        # Save the original input audio as source for voice extraction
                        original_audio_path = temp_dir / "original_audio.wav"
                        original_audio_np = audio_tensor.squeeze().cpu().numpy()
                        
                        # Check if original audio is valid
                        if len(original_audio_np) < 8000:  # Less than 0.5 seconds at 16kHz
                            logger.warning(f"Original audio too short for voice cloning: {len(original_audio_np)} samples")
                            raise ValueError("Original audio too short for voice cloning")
                            
                        # Save original audio as WAV for OpenVoice reference
                        sf.write(str(original_audio_path), original_audio_np, 16000)
                        logger.info(f"Saved original audio for voice extraction: {os.path.getsize(str(original_audio_path))} bytes")
                        
                        # Call the voice cloning function with the original audio and the translated text
                        voice_cloning_success = self._clone_voice_with_api(
                            input_audio_path=str(original_audio_path),
                            target_text=target_text,  # Use the actual translated text here
                            output_audio_path=str(output_path)
                        )
                        
                        if voice_cloning_success:
                            # Load the converted audio
                            y_converted, sr_converted = sf.read(str(output_path))
                            output_tensor = torch.FloatTensor(y_converted).unsqueeze(0)
                            
                            logger.info(f"Voice cloning successful, output shape: {output_tensor.shape}")
                        else:
                            # Fallback to base audio if voice cloning failed
                            logger.warning("Voice cloning failed, falling back to base audio")
                            y, sr = sf.read(str(base_wav_path))
                            output_tensor = torch.FloatTensor(y).unsqueeze(0)
                    except Exception as e:
                        logger.error(f"Voice cloning error: {str(e)}")
                        logger.warning("Falling back to base audio due to error")
                        # Load the base audio as fallback
                        y, sr = sf.read(str(base_wav_path))
                        output_tensor = torch.FloatTensor(y).unsqueeze(0)
                            
                    except Exception as e:
                        logger.error(f"Voice cloning via API failed: {str(e)}")
                        logger.warning("Falling back to base audio")
                        
                        # Fall back to base audio
                        y, sr = sf.read(str(base_wav_path))
                        output_tensor = torch.FloatTensor(y).unsqueeze(0)
                
                # Return results
                duration = time.time() - start_time
                logger.info(f"Translation completed in {duration:.2f}s")
                
                return {
                    "audio": output_tensor,
                    "transcripts": {
                        "source": source_text,
                        "target": target_text
                    }
                }
                    
            except Exception as e:
                logger.error(f"Translation failed: {str(e)}")
                
                # Create a silent audio response as fallback
                silent_audio = torch.zeros((1, 16000))  # 1 second of silence
                
                return {
                    "audio": silent_audio,
                    "transcripts": {
                        "source": "Error occurred during translation",
                        "target": "Error occurred during translation"
                    }
                }