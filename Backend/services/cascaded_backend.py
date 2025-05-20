# services/cascaded_backend.py
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import soundfile as sf
import librosa
import time
import traceback
from pydub import AudioSegment
import requests 

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- AWS Polly (boto3) Import ---
try:
    import boto3
    BOTO3_AVAILABLE = True
    logger_polly = logging.getLogger(__name__)
    logger_polly.info("AWS SDK (boto3) FOUND and imported successfully.")
except ImportError:
    logger_polly = logging.getLogger(__name__)
    logger_polly.warning("AWS SDK (boto3) NOT FOUND. AWS Polly TTS will not be available.")
    BOTO3_AVAILABLE = False
    boto3 = None 
# --- End AWS Polly Import ---

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers(): 
    import sys
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False 
    logger.warning("Root logger has no handlers. CascadedBackend is setting up its own console logger.")
else:
    logger.setLevel(logging.INFO)

try:
    import whisper
    WHISPER_AVAILABLE_FLAG = True 
    logger.info("Whisper library FOUND and imported successfully.")
except ImportError:
    logger.warning("Whisper library NOT FOUND. ASR will use transformers pipeline as fallback.")
    WHISPER_AVAILABLE_FLAG = False

def check_openvoice_api():
    try:
        response = requests.get("http://localhost:8000/status", timeout=2)
        if response.status_code == 200: logger.info("OpenVoice API is available."); return True
        logger.warning(f"OpenVoice API status check failed: {response.status_code}"); return False
    except Exception as e: logger.warning(f"OpenVoice API not available: {e}"); return False
OPENVOICE_API_AVAILABLE = check_openvoice_api()
    
from .translation_strategy import TranslationBackend

class CascadedBackend(TranslationBackend):
    def __init__(self, device=None, use_voice_cloning=True):
        logger.info(f"Initializing CascadedBackend with device: {device}, use_voice_cloning_config: {use_voice_cloning}")
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"CascadedBackend will use device: {self.device}")
        self.initialized = False
        self.use_voice_cloning_config = use_voice_cloning 
        
        # App's internal language codes (e.g., 'eng', 'deu') mapped to Polly VoiceID, Engine, and Region
        # Find Voice IDs & Regions: https://docs.aws.amazon.com/polly/latest/dg/voicelist.html
        # Neural voices are generally higher quality.
        self.polly_voice_map = {
            'eng': {'VoiceId': 'Joanna', 'Engine': 'neural', 'Region': 'us-east-1'},  # US English
            'deu': {'VoiceId': 'Vicki',  'Engine': 'neural', 'Region': 'eu-central-1'}, # German
            'spa': {'VoiceId': 'Lucia',  'Engine': 'neural', 'Region': 'eu-west-1'},   # Castilian Spanish
            'fra': {'VoiceId': 'Lea',    'Engine': 'neural', 'Region': 'eu-west-1'},   # French
            'ita': {'VoiceId': 'Bianca', 'Engine': 'neural', 'Region': 'eu-central-1'},# Italian
            'jpn': {'VoiceId': 'Mizuki', 'Engine': 'standard', 'Region': 'ap-northeast-1'},# Japanese (Standard, check for Neural if avail)
            'kor': {'VoiceId': 'Seoyeon','Engine': 'neural', 'Region': 'ap-northeast-2'},# Korean
            'por': {'VoiceId': 'Camila', 'Engine': 'neural', 'Region': 'sa-east-1'},   # Brazilian Portuguese
            'rus': {'VoiceId': 'Tatyana','Engine': 'standard', 'Region': 'eu-central-1'},# Russian
            'cmn': {'VoiceId': 'Zhiyu',  'Engine': 'neural', 'Region': 'ap-northeast-1'},# Mandarin Chinese
            'ara': {'VoiceId': 'Zeina',  'Engine': 'standard', 'Region': 'me-south-1'},  # Arabic
            'hin': {'VoiceId': 'Aditi',  'Engine': 'standard', 'Region': 'ap-south-1'},  # Hindi (Indian English accent)
            'nld': {'VoiceId': 'Laura',  'Engine': 'neural', 'Region': 'eu-west-1'},   # Dutch
            'pol': {'VoiceId': 'Ewa',    'Engine': 'neural', 'Region': 'eu-central-1'},  # Polish
            'tur': {'VoiceId': 'Filiz',  'Engine': 'standard', 'Region': 'eu-central-1'},# Turkish
            'ukr': {'VoiceId': 'Tatyana','Engine': 'standard', 'Region': 'eu-central-1'}, # No dedicated Ukrainian, use Russian as fallback
            # Add more languages and ensure VoiceId/Engine/Region are correct from Polly docs
        }
        
        # For gTTS fallback and Whisper ASR language hint (maps app codes 'deu' -> 'de')
        self.simple_lang_code_map = { 
            'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it', 'por': 'pt', 
            'cmn': 'zh-cn', 'jpn': 'ja', 'kor': 'ko', 'ara': 'ar', 'hin': 'hi', 
            'nld': 'nl', 'rus': 'ru', 'pol': 'pl', 'tur': 'tr', 'ukr': 'uk'
            # Add any other app codes you use that need mapping to simple ISO 639-1
        }
        self.display_language_names = { # For get_supported_languages to show in UI
            'eng': 'English (US - Joanna)', 'fra': 'French (Lea)', 'spa': 'Spanish (Lucia)', 
            'deu': 'German (Vicki)', 'ita': 'Italian (Bianca)', 'jpn': 'Japanese (Mizuki)',
            'kor': 'Korean (Seoyeon)', 'por': 'Portuguese (BR - Camila)', 'rus': 'Russian (Tatyana)',
            'cmn': 'Chinese (Mandarin - Zhiyu)', 'ara': 'Arabic (Zeina)', 'hin': 'Hindi (Aditi)',
            'nld': 'Dutch (Laura)', 'pol': 'Polish (Ewa)', 'tur': 'Turkish (Filiz)',
            'ukr': 'Ukrainian (Fallback Tatyana/Russian Voice)'
        }

        self.asr_model = None; self.asr_pipeline = None
        self.translator_model = None; self.translator_tokenizer = None
        self.polly_clients_cache = {} # Cache Polly clients per region

        logger.info(f"CascadedBackend __init__. Whisper: {WHISPER_AVAILABLE_FLAG}, OpenVoice API: {OPENVOICE_API_AVAILABLE}, AWS Polly (boto3): {BOTO3_AVAILABLE}")
    
    def _get_polly_client(self, region_name: str):
        if not BOTO3_AVAILABLE: logger.warning("boto3 not available."); return None
        if region_name in self.polly_clients_cache:
            return self.polly_clients_cache[region_name]
        try:
            logger.info(f"Creating AWS Polly client for region: {region_name}")
            client = boto3.client('polly', region_name=region_name)
            self.polly_clients_cache[region_name] = client
            logger.info(f"AWS Polly client for {region_name} created successfully.")
            return client
        except Exception as e:
            logger.error(f"Failed to create AWS Polly client for region {region_name}: {e}", exc_info=True)
            return None

    def initialize(self):
        if self.initialized: logger.info("CascadedBackend already initialized."); return
        try:
            logger.info(f"Initializing CascadedBackend on device: {self.device}"); start_time = time.time()
            if WHISPER_AVAILABLE_FLAG: # ASR
                self.asr_model = whisper.load_model("medium", device=self.device); logger.info("Whisper ASR loaded.")
            else: self.asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if self.device.type == 'cuda' and torch.cuda.is_available() else -1); logger.info("Transformers ASR loaded.")
            
            # NLLB Translation
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M"); self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            if self.device.type == 'cuda' and torch.cuda.is_available(): self.translator_model = self.translator_model.to(self.device)
            logger.info("NLLB translation model and tokenizer loaded.")

            # Initialize Polly client for a default/common region to check credentials early
            if BOTO3_AVAILABLE: self._get_polly_client(os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

            self.initialized = True; logger.info(f"CascadedBackend initialized in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize CascadedBackend: {e}", exc_info=True); self.initialized = False; raise 
    
    def _convert_to_nllb_code(self, lang_code_app: str) -> str:
        mapping = { 'eng':'eng_Latn', 'fra':'fra_Latn', 'spa':'spa_Latn', 'deu':'deu_Latn', 'ita':'ita_Latn', 
                    'por':'por_Latn', 'rus':'rus_Cyrl', 'cmn':'zho_Hans', 'jpn':'jpn_Jpan', 'kor':'kor_Hang', 
                    'ara':'ara_Arab', 'hin':'hin_Deva', 'nld':'nld_Latn', 'pol':'pol_Latn', 'tur':'tur_Latn', 
                    'ukr':'ukr_Cyrl', 'ces':'ces_Latn', 'hun':'hun_Latn'}
        return mapping.get(lang_code_app, 'eng_Latn')

    def _get_simple_lang_code(self, lang_code_app: str) -> str: 
        return self.simple_lang_code_map.get(lang_code_app, 'en')

    def _generate_base_tts_audio_with_polly(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        polly_voice_config = self.polly_voice_map.get(lang_code_app)
        if not polly_voice_config:
            logger.warning(f"No AWS Polly voice config for app lang '{lang_code_app}'. Cannot use Polly.")
            return None

        polly_client = self._get_polly_client(polly_voice_config['Region'])
        if not polly_client:
            logger.warning(f"AWS Polly client for region '{polly_voice_config['Region']}' not available.")
            return None
        
        logger.info(f"Attempting Polly TTS for '{lang_code_app}' with VoiceId '{polly_voice_config['VoiceId']}'.")
        output_mp3_path = temp_audio_dir / f"base_tts_polly_{lang_code_app}.mp3"
        final_wav_path = temp_audio_dir / f"base_tts_polly_{lang_code_app}_16k.wav"

        try:
            response = polly_client.synthesize_speech(
                Text=text, OutputFormat='mp3', VoiceId=polly_voice_config['VoiceId'], Engine=polly_voice_config['Engine']
            )
            if 'AudioStream' in response:
                with open(output_mp3_path, 'wb') as f: f.write(response['AudioStream'].read())
                if not output_mp3_path.exists() or output_mp3_path.stat().st_size < 100:
                    logger.error("Polly MP3 output small/missing."); return None
                logger.info(f"Polly MP3 OK: {output_mp3_path} ({output_mp3_path.stat().st_size}b)")
                
                sound = AudioSegment.from_mp3(str(output_mp3_path))
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(str(final_wav_path), format="wav")
                if not final_wav_path.exists() or final_wav_path.stat().st_size < 1000:
                    logger.error("Polly MP3->WAV conversion failed."); return None
                logger.info(f"Polly 16kHz WAV OK: {final_wav_path}")
                try: os.remove(output_mp3_path)
                except: pass
                return final_wav_path
            else: logger.error(f"Polly response no AudioStream: {response}"); return None
        except Exception as e: logger.error(f"Polly TTS failed for '{lang_code_app}': {e}", exc_info=True); return None

    def _generate_base_tts_audio(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        logger.info(f"[_generate_base_tts_audio] For app_lang '{lang_code_app}', text: '{text[:70]}...'")
        
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map:
            polly_audio_path = self._generate_base_tts_audio_with_polly(text, lang_code_app, temp_audio_dir)
            if polly_audio_path: return polly_audio_path
            logger.warning(f"AWS Polly failed for '{lang_code_app}', trying gTTS fallback...")
        else: logger.info(f"AWS Polly not configured/available for '{lang_code_app}'. Trying gTTS fallback.")

        logger.warning(f"Falling back to gTTS for language '{lang_code_app}'.")
        # ... (gTTS logic from your last working version, ensure unique output path for gTTS) ...
        base_tts_gtts_output_wav_path = temp_audio_dir / f"base_tts_gtts_{lang_code_app}.wav"
        try: from gtts import gTTS
        except ImportError: logger.error("gTTS library not installed."); return None
        gtts_lang_code = self._get_simple_lang_code(lang_code_app)
        base_tts_mp3_path = temp_audio_dir / f"base_tts_{lang_code_app}_gtts_fallback.mp3"
        try:
            tts_g = gTTS(text=text, lang=gtts_lang_code, slow=False)
            tts_g.save(str(base_tts_mp3_path))
            if not base_tts_mp3_path.exists() or base_tts_mp3_path.stat().st_size == 0: logger.error(f"gTTS MP3 gen failed."); return None
            sound = AudioSegment.from_mp3(str(base_tts_mp3_path))
            sound = sound.set_frame_rate(16000).set_channels(1)
            sound.export(str(base_tts_gtts_output_wav_path), format="wav")
            if not base_tts_gtts_output_wav_path.exists() or base_tts_gtts_output_wav_path.stat().st_size < 1000: logger.error(f"gTTS WAV conversion failed."); return None
            logger.info(f"gTTS (fallback) WAV audio generated: {base_tts_gtts_output_wav_path}");
            try: os.remove(base_tts_mp3_path)
            except OSError: pass
            return base_tts_gtts_output_wav_path
        except Exception as e_gtts: logger.error(f"gTTS fallback failed: {e_gtts}", exc_info=True); return None

        logger.error(f"All TTS options failed for language '{lang_code_app}'.")
        return None

    # _clone_voice_with_api, translate_speech methods remain the same as message #48 (the one with MeloTTS that you said was good apart from German)
    # Ensure translate_speech calls _generate_base_tts_audio correctly.

    def _clone_voice_with_api(self, ref_voice_audio_path, target_content_audio_path, output_audio_path):
        logger.info(f"[_clone_voice_with_api] Calling OpenVoice API.")
        if not OPENVOICE_API_AVAILABLE: logger.warning("OpenVoice API NOT available."); return False
        if not Path(ref_voice_audio_path).exists() or Path(ref_voice_audio_path).stat().st_size < 1000:
            logger.error(f"Ref voice MISSING/small: {ref_voice_audio_path}"); return False
        if not Path(target_content_audio_path).exists() or Path(target_content_audio_path).stat().st_size < 1000:
            logger.error(f"Target content audio (TTS) MISSING/small: {target_content_audio_path}"); return False
        try:
            with open(ref_voice_audio_path, "rb") as f_ref, open(target_content_audio_path, "rb") as f_target_content:
                files = {"audio_file": (Path(ref_voice_audio_path).name, f_ref, "audio/wav"),
                         "target_file": (Path(target_content_audio_path).name, f_target_content, "audio/wav")}
                response = requests.post("http://localhost:8000/clone-voice", files=files, timeout=180)
            logger.info(f"OpenVoice API status: {response.status_code}")
            if response.status_code == 200:
                with open(output_audio_path, "wb") as f_out: f_out.write(response.content)
                if Path(output_audio_path).exists() and Path(output_audio_path).stat().st_size > 1000:
                    logger.info(f"OpenVoice cloning OK: {output_audio_path}"); return True
                logger.error(f"OpenVoice output file problematic: {output_audio_path}"); return False
            logger.error(f"OpenVoice API error: {response.status_code} - {response.text[:200]}"); return False
        except Exception as e: logger.error(f"Exception in OpenVoice API call: {e}", exc_info=True); return False
        
    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str = "eng", target_lang: str = "fra") -> Dict[str, Any]:
        logger.info(f"[translate_speech] CALLED. app_source_lang='{source_lang}', app_target_lang='{target_lang}'")
        if not self.initialized: self.initialize()
        
        start_time = time.time()
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_text, target_text = "ASR_FAILED", "TRANSLATION_FAILED"
            output_tensor = torch.zeros((1, 16000), dtype=torch.float32) 

            try:
                audio_numpy = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
                original_audio_ref_path = temp_dir / "original_audio_ref_for_cloning.wav"
                sf.write(str(original_audio_ref_path), audio_numpy, 16000)

                if np.abs(audio_numpy).max() < 1e-5: source_text = "" 
                elif WHISPER_AVAILABLE_FLAG and self.asr_model:
                    whisper_lang_hint = self._get_simple_lang_code(source_lang)
                    asr_result = self.asr_model.transcribe(audio_numpy, language=whisper_lang_hint, task="transcribe", fp16=(self.device.type == 'cuda' and torch.cuda.is_available()))
                    source_text = asr_result["text"]; logger.info(f"Whisper ASR: '{source_text[:70]}...'")
                elif self.asr_pipeline:
                    temp_asr_path = temp_dir / "asr_input.wav"; sf.write(str(temp_asr_path), audio_numpy, 16000)
                    source_text = self.asr_pipeline(str(temp_asr_path))["text"]; logger.info(f"Transformers ASR: '{source_text[:70]}...'")
                else: logger.error("No ASR model available.")

                if not source_text or source_text == "ASR_FAILED": target_text = ""
                else:
                    src_nllb_code = self._convert_to_nllb_code(source_lang); tgt_nllb_code = self._convert_to_nllb_code(target_lang)
                    input_ids = self.translator_tokenizer(source_text, return_tensors="pt", padding=True).input_ids.to(self.device)
                    forced_bos_id = None 
                    if hasattr(self.translator_tokenizer, 'get_lang_id'):
                        try: forced_bos_id = self.translator_tokenizer.get_lang_id(tgt_nllb_code)
                        except Exception: 
                            forced_bos_id = self.translator_tokenizer.convert_tokens_to_ids(tgt_nllb_code)
                            if forced_bos_id == self.translator_tokenizer.unk_token_id: forced_bos_id = None
                    elif hasattr(self.translator_tokenizer, 'lang_code_to_id') and tgt_nllb_code in self.translator_tokenizer.lang_code_to_id:
                        forced_bos_id = self.translator_tokenizer.lang_code_to_id[tgt_nllb_code]
                    if forced_bos_id is None: logger.warning(f"NLLB BOS for '{tgt_nllb_code}' not found.")
                    gen_kwargs = {"input_ids": input_ids, "max_length": 1024, "num_beams": 5, "length_penalty": 1.0}
                    if forced_bos_id is not None: gen_kwargs["forced_bos_token_id"] = forced_bos_id
                    with torch.no_grad(): translated_tokens = self.translator_model.generate(**gen_kwargs)
                    target_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                logger.info(f"NLLB Translation: '{target_text[:70]}...'")
                
                base_tts_path = None
                if not target_text.strip(): logger.warning("Target text empty, TTS output will be silent.")
                else: base_tts_path = self._generate_base_tts_audio(target_text, target_lang, temp_dir)
                
                if base_tts_path and base_tts_path.exists():
                    y_base, _ = sf.read(str(base_tts_path)); output_tensor = torch.FloatTensor(y_base.astype(np.float32)).unsqueeze(0)
                else: logger.error("Base TTS audio generation failed or text was empty. Using silence.")
                
                should_clone = self.use_voice_cloning_config and OPENVOICE_API_AVAILABLE and base_tts_path is not None
                logger.info(f"OpenVoice cloning decision: Config={self.use_voice_cloning_config}, API={OPENVOICE_API_AVAILABLE}, BaseTTS_OK={base_tts_path is not None} -> Clone={should_clone}")
                if should_clone:
                    cloned_final_path = temp_dir / "cloned_final_audio.wav"
                    clone_success = self._clone_voice_with_api(str(original_audio_ref_path), str(base_tts_path), str(cloned_final_path))
                    if clone_success and cloned_final_path.exists() and cloned_final_path.stat().st_size > 1000:
                        y_cloned, _ = sf.read(str(cloned_final_path)); output_tensor = torch.FloatTensor(y_cloned.astype(np.float32)).unsqueeze(0)
                        logger.info(f"Using OpenVoice CLONED audio.")
                    else: logger.warning("OpenVoice cloning FAILED. Using base TTS audio.")
                else: logger.info("OpenVoice cloning SKIPPED.")
                logger.info(f"Final audio shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
            except Exception as e: logger.error(f"Error in translate_speech pipeline: {e}", exc_info=True)
            logger.info(f"Processing completed in {time.time() - start_time:.2f}s")            
            return {"audio": output_tensor, "transcripts": {"source": source_text, "target": target_text}}

    # --- Implementation of Abstract Methods ---
    def is_language_supported(self, lang_code_app: str) -> bool:
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map: return True
        # Add MeloTTS check here if you re-integrate it as a primary option
        # if MELOTTS_AVAILABLE and lang_code_app in self.melo_lang_map: ... return True ...
        if lang_code_app in self.simple_lang_code_map: return True # gTTS fallback
        logger.warning(f"[is_language_supported] '{lang_code_app}' not in Polly map or gTTS map.")
        return False
    
    def get_supported_languages(self) -> Dict[str, str]:
        supported_for_display = {}
        if BOTO3_AVAILABLE:
            for app_code, polly_config in self.polly_voice_map.items():
                display_name = self.display_language_names.get(app_code, f"{app_code.upper()} (Polly: {polly_config['VoiceId']})")
                supported_for_display[app_code] = display_name
        
        # If you add MeloTTS back as a primary, merge its languages here, prioritizing Polly if a lang is in both
        # For gTTS fallback for languages not covered by Polly:
        for app_code in self.simple_lang_code_map.keys():
            if app_code not in supported_for_display:
                 supported_for_display[app_code] = self.display_language_names.get(app_code, app_code.upper()) + " (gTTS fallback)"
        
        if not supported_for_display: # Absolute fallback
             return {k: v + " (gTTS)" for k,v in self.display_language_names.items() if k in self.simple_lang_code_map}
        logger.debug(f"[get_supported_languages] Returning: {supported_for_display}")
        return supported_for_display