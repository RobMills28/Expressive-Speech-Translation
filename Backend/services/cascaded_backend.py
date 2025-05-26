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
from pydub import AudioSegment # For MP3 to WAV conversion
import requests
import json
import sys
import uuid # For unique IDs

# --- AWS Polly (boto3) Import with DETAILED logging ---
BOTO3_AVAILABLE = False
boto3 = None
# This logger will inherit settings from the root logger configured in app.py
# Its level is effectively controlled by app.py's root logger and specific library settings
polly_init_logger = logging.getLogger(f"{__name__}.boto3_init_check")


try:
    polly_init_logger.info("Attempting to import 'boto3' for AWS Polly SDK...")
    import boto3 # type: ignore
    polly_init_logger.info(f"Successfully imported 'boto3'. Version: {getattr(boto3, '__version__', 'unknown')}")

    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1") # Ensure this is your desired default
    polly_init_logger.info(f"Attempting to create a test Polly client in region: '{aws_default_region}'...")
    test_client = boto3.client('polly', region_name=aws_default_region)
    polly_init_logger.info("Testing Polly client with 'describe_voices'...")
    test_client.describe_voices(LanguageCode='en-US') # Example call to verify client
    polly_init_logger.info(f"Boto3 Polly client test successful in region '{aws_default_region}'. Credentials & region seem OK.")
    BOTO3_AVAILABLE = True
except ImportError as e_import:
    polly_init_logger.error(f"IMPORT ERROR: Failed to import 'boto3' - {e_import}. AWS Polly will be unavailable.", exc_info=False)
    BOTO3_AVAILABLE = False
    boto3 = None
except Exception as e_client_or_general:
    polly_init_logger.error(f"BOTO3 CLIENT FAILURE: 'boto3' imported, but Polly client creation/test FAILED: {type(e_client_or_general).__name__} - {e_client_or_general}", exc_info=False)
    polly_init_logger.error("DETAILS: This usually means AWS credentials (e.g., from `aws configure` or IAM role) "
                           "and/or AWS_DEFAULT_REGION are not correctly configured for this environment, or network/IAM permission issues.")
    BOTO3_AVAILABLE = False
    if 'boto3' not in sys.modules:
        boto3 = None
# --- End AWS Polly Import ---

logger = logging.getLogger(__name__) # Main logger for this module

try:
    import whisper # type: ignore
    WHISPER_AVAILABLE_FLAG = True
    logger.info("Whisper library FOUND and imported for CascadedBackend.")
except ImportError:
    logger.warning("Whisper library NOT FOUND for CascadedBackend. Consider installing it for better ASR. Falling back to Transformers pipeline.")
    WHISPER_AVAILABLE_FLAG = False
    whisper = None

OPENVOICE_API_URL = os.getenv("OPENVOICE_API_URL", "http://localhost:8000")
OPENVOICE_API_AVAILABLE = False

def check_openvoice_api():
    global OPENVOICE_API_AVAILABLE # Allow modification of the global flag
    try:
        status_url = f"{OPENVOICE_API_URL}/status"
        logger.debug(f"Checking OpenVoice API status at: {status_url}")
        response = requests.get(status_url, timeout=3) # Short timeout for status check
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("tone_converter_model_loaded") and data.get("default_source_se_loaded"):
                    logger.info(f"OpenVoice API available and models loaded: {data.get('message', 'Status OK')}")
                    OPENVOICE_API_AVAILABLE = True
                    return True
                else:
                    logger.warning(f"OpenVoice API is UP ({response.status_code}), but models are NOT fully loaded: {data}")
                    OPENVOICE_API_AVAILABLE = False
                    return False
            except json.JSONDecodeError:
                logger.warning(f"OpenVoice API /status returned 200 but response was not valid JSON: {response.text[:200]}")
                OPENVOICE_API_AVAILABLE = False
                return False
        elif response.status_code == 503: # Service Unavailable
            logger.warning(f"OpenVoice API /status returned 503: Service unavailable. Detail: {response.text[:200]}")
            OPENVOICE_API_AVAILABLE = False
            return False
        else:
            logger.warning(f"OpenVoice API status check failed with HTTP {response.status_code}. Detail: {response.text[:200]}")
            OPENVOICE_API_AVAILABLE = False
            return False
    except requests.exceptions.RequestException as e_req: # Catches network errors, timeouts, etc.
        logger.warning(f"OpenVoice API unavailable (RequestException): {type(e_req).__name__} - {e_req}")
        OPENVOICE_API_AVAILABLE = False
        return False
    except Exception as e_gen: # Catch any other unexpected errors
        logger.warning(f"OpenVoice API status check encountered an unexpected error: {type(e_gen).__name__} - {e_gen}")
        OPENVOICE_API_AVAILABLE = False
        return False

check_openvoice_api() # Initial check on module load

from .translation_strategy import TranslationBackend
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline

class CascadedBackend(TranslationBackend):
    def __init__(self, device=None, use_voice_cloning=True):
        logger.info(f"Initializing CascadedBackend. Device: {device}, Voice Cloning Config: {use_voice_cloning}, PollySDK Flag from import: {BOTO3_AVAILABLE}")
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"CascadedBackend will use ML device: {self.device}")
        self.initialized = False
        self.use_voice_cloning_config = use_voice_cloning

        self.polly_voice_map = {'eng':{'VoiceId':'Joanna','Engine':'neural','Region':'us-east-1'},'deu':{'VoiceId':'Vicki','Engine':'neural','Region':'eu-central-1'},'spa':{'VoiceId':'Lucia','Engine':'neural','Region':'eu-west-1'},'fra':{'VoiceId':'Lea','Engine':'neural','Region':'eu-west-1'},'ita':{'VoiceId':'Bianca','Engine':'neural','Region':'eu-central-1'},'jpn':{'VoiceId':'Kazuha','Engine':'neural','Region':'ap-northeast-1'},'kor':{'VoiceId':'Seoyeon','Engine':'neural','Region':'ap-northeast-2'},'por':{'VoiceId':'Ines','Engine':'neural','Region':'eu-west-1'},'rus':{'VoiceId':'Olga','Engine':'neural','Region':'eu-central-1'},'cmn':{'VoiceId':'Zhiyu','Engine':'neural','Region':'ap-northeast-1'},'ara':{'VoiceId':'Hala','Engine':'neural','Region':'me-south-1'},'hin':{'VoiceId':'Kajal','Engine':'neural','Region':'ap-south-1'},'nld':{'VoiceId':'Laura','Engine':'neural','Region':'eu-west-1'},'pol':{'VoiceId':'Ewa','Engine':'neural','Region':'eu-central-1'},'tur':{'VoiceId':'Burcu','Engine':'neural','Region':'eu-central-1'},'ukr':{'VoiceId':'Olga','Engine':'neural','Region':'eu-central-1'}}
        self.simple_lang_code_map = { 'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it', 'por': 'pt', 'cmn': 'zh-cn', 'jpn': 'ja', 'kor': 'ko', 'ara': 'ar', 'hin': 'hi', 'nld': 'nl', 'rus': 'ru', 'pol': 'pl', 'tur': 'tr', 'ukr': 'uk'}
        self.display_language_names = {'eng':'English (Joanna, Neural)','deu':'German (Vicki, Neural)','spa':'Spanish (Lucia, Neural)','fra':'French (Lea, Neural)','ita':'Italian (Bianca, Neural)','jpn':'Japanese (Kazuha, Neural)','kor':'Korean (Seoyeon, Neural)','por':'Portuguese (Ines, Eur., Neural)','rus':'Russian (Olga, Neural)','cmn':'Mandarin (Zhiyu, Neural)','ara':'Arabic (Hala, Egy., Neural)','hin':'Hindi (Kajal, Neural)','nld':'Dutch (Laura, Neural)','pol':'Polish (Ewa, Neural)','tur':'Turkish (Burcu, Neural)','ukr':'Ukrainian (Fallback Olga, Rus., Neural)'}

        self.asr_model = None; self.asr_pipeline = None; self.translator_model = None; self.translator_tokenizer = None; self.polly_clients_cache = {}
        logger.info(f"CascadedBackend __init__ final flags check: WhisperImported={WHISPER_AVAILABLE_FLAG}, OpenVoiceAPIUp={OPENVOICE_API_AVAILABLE}, PollySDKImported={BOTO3_AVAILABLE}")

    def _get_polly_client(self, region_name: str):
        if not BOTO3_AVAILABLE or not boto3: # Check if boto3 was successfully imported
            logger.warning("Boto3 SDK not available or not imported, cannot create Polly client.")
            return None
        if region_name in self.polly_clients_cache:
            logger.debug(f"Returning cached Polly client for region: {region_name}")
            return self.polly_clients_cache[region_name]
        try:
            logger.info(f"Creating new Polly client for region: {region_name}")
            client = boto3.client('polly', region_name=region_name)
            # Optional: Test client immediately, e.g., client.describe_voices(MaxResults=1)
            self.polly_clients_cache[region_name] = client
            logger.info(f"Polly client for {region_name} created and cached successfully.")
            return client
        except Exception as e:
            logger.error(f"Failed to create Polly client for region {region_name}: {e}", exc_info=True)
            return None

    def initialize(self):
        if self.initialized:
            logger.info("CascadedBackend already initialized.")
            return
        try:
            logger.info(f"Initializing CascadedBackend components on device: {self.device}")
            start_time = time.time()
            if WHISPER_AVAILABLE_FLAG and whisper:
                self.asr_model = whisper.load_model("medium", device=self.device)
                logger.info("Whisper ASR loaded ('medium').")
            else:
                logger.info("Using Hugging Face Transformers ASR pipeline as Whisper library is not available.")
                self.asr_pipeline = hf_pipeline("automatic-speech-recognition",
                                                model="openai/whisper-base",
                                                device=0 if self.device.type == 'cuda' else -1) # device 0 for cuda, -1 for cpu
                logger.info("Transformers ASR pipeline loaded ('openai/whisper-base').")

            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            if self.device.type == 'cuda' and torch.cuda.is_available():
                self.translator_model = self.translator_model.to(self.device)
            logger.info("NLLB translation model and tokenizer loaded.")

            if BOTO3_AVAILABLE:
                default_aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
                self._get_polly_client(default_aws_region) # Pre-cache client for default region

            self.initialized = True
            logger.info(f"CascadedBackend initialized successfully in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize CascadedBackend: {e}", exc_info=True)
            self.initialized = False
            raise

    def _convert_to_nllb_code(self, lang_code_app: str) -> str:
        mapping = {'eng':'eng_Latn','fra':'fra_Latn','spa':'spa_Latn','deu':'deu_Latn','ita':'ita_Latn','por':'por_Latn','rus':'rus_Cyrl','cmn':'zho_Hans','jpn':'jpn_Jpan','kor':'kor_Hang','ara':'ara_Arab','hin':'hin_Deva','nld':'nld_Latn','pol':'pol_Latn','tur':'tur_Latn','ukr':'ukr_Cyrl','ces':'ces_Latn','hun':'hun_Latn'}
        return mapping.get(lang_code_app.lower(), 'eng_Latn')

    def _get_simple_lang_code(self, lang_code_app: str) -> str:
        return self.simple_lang_code_map.get(lang_code_app.lower(), 'en')

    def _generate_base_tts_audio_with_polly(self, text: str, lang_code_app: str, temp_audio_dir: Path, use_ssml: bool = False) -> Optional[Path]:
        polly_voice_config = self.polly_voice_map.get(lang_code_app)
        if not polly_voice_config:
            logger.warning(f"No Polly voice configuration for app language code '{lang_code_app}'.")
            return None

        polly_client = self._get_polly_client(polly_voice_config['Region'])
        if not polly_client:
            logger.warning(f"Polly client for region '{polly_voice_config['Region']}' unavailable for '{lang_code_app}'.")
            return None

        logger.info(f"Attempting Polly TTS for '{lang_code_app}', VoiceId '{polly_voice_config['VoiceId']}', Engine '{polly_voice_config['Engine']}'. SSML: {use_ssml}")
        request_id_tts = str(uuid.uuid4())[:8]
        output_mp3_path = temp_audio_dir / f"base_polly_{lang_code_app}_{request_id_tts}.mp3"
        final_wav_path = temp_audio_dir / f"base_polly_{lang_code_app}_{request_id_tts}_16k.wav"

        try:
            req_params = {'Text': text, 'OutputFormat': 'mp3', 'VoiceId': polly_voice_config['VoiceId'], 'Engine': polly_voice_config['Engine']}
            if use_ssml:
                req_params['TextType'] = 'ssml'
                if not text.strip().lower().startswith("<speak>"): req_params['Text'] = f"<speak>{text}</speak>"
            logger.debug(f"Polly synthesize_speech request params: {req_params}")
            resp = polly_client.synthesize_speech(**req_params)

            if 'AudioStream' in resp:
                with open(output_mp3_path, 'wb') as f: f.write(resp['AudioStream'].read())
                if not output_mp3_path.exists() or output_mp3_path.stat().st_size < 100:
                    logger.error(f"Polly MP3 output for '{lang_code_app}' missing/small. Size: {output_mp3_path.stat().st_size if output_mp3_path.exists() else 'Not found'}")
                    return None
                
                sound = AudioSegment.from_mp3(str(output_mp3_path)).set_frame_rate(16000).set_channels(1)
                sound.export(str(final_wav_path), format="wav")
                
                if not final_wav_path.exists() or final_wav_path.stat().st_size < 1000:
                    logger.error(f"Polly WAV conversion for '{lang_code_app}' failed or output small. Size: {final_wav_path.stat().st_size if final_wav_path.exists() else 'Not found'}")
                    return None
                logger.info(f"Polly 16kHz WAV OK for '{lang_code_app}': {final_wav_path.name} (Size: {final_wav_path.stat().st_size}b)")
                try: os.remove(output_mp3_path)
                except OSError: pass
                return final_wav_path
            else:
                logger.error(f"Polly API call for '{lang_code_app}' did not return 'AudioStream'. Response: {resp}")
                return None
        except boto3.exceptions.Boto3Error as e_boto:
             logger.error(f"Boto3/Polly API Error for '{lang_code_app}': {e_boto}", exc_info=False)
             logger.error(f"Details: Type={type(e_boto).__name__}, Args={e_boto.args}")
             if hasattr(e_boto, 'response') and e_boto.response and 'Error' in e_boto.response: # type: ignore
                 logger.error(f"AWS Error Code: {e_boto.response['Error'].get('Code')}, Message: {e_boto.response['Error'].get('Message')}") # type: ignore
             return None
        except Exception as e:
            logger.error(f"Generic Polly TTS generation failure for '{lang_code_app}': {e}", exc_info=True)
            return None

    def _generate_base_tts_audio_with_gtts(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        logger.warning(f"Using Fallback gTTS for language '{lang_code_app}'. Text: '{text[:70]}...'")
        try:
            from gtts import gTTS # type: ignore
        except ImportError:
            logger.error("gTTS library not installed. Fallback TTS unavailable.")
            return None

        gtts_lang_code = self._get_simple_lang_code(lang_code_app)
        request_id_tts = str(uuid.uuid4())[:8]
        output_mp3_path = temp_audio_dir / f"base_gtts_{lang_code_app}_{request_id_tts}_fb.mp3"
        final_wav_path = temp_audio_dir / f"base_gtts_{lang_code_app}_{request_id_tts}_16k_fb.wav"

        try:
            tts = gTTS(text=text, lang=gtts_lang_code, slow=False)
            tts.save(str(output_mp3_path))
            if not output_mp3_path.exists() or output_mp3_path.stat().st_size == 0:
                logger.error(f"gTTS MP3 generation for '{lang_code_app}' failed (file missing/empty).")
                return None
            sound = AudioSegment.from_mp3(str(output_mp3_path)).set_frame_rate(16000).set_channels(1)
            sound.export(str(final_wav_path), format="wav")
            if not final_wav_path.exists() or final_wav_path.stat().st_size < 1000:
                logger.error(f"gTTS WAV conversion for '{lang_code_app}' failed or output small.")
                return None
            logger.info(f"gTTS 16kHz WAV for '{lang_code_app}' generated: {final_wav_path.name}")
            try: os.remove(output_mp3_path)
            except OSError: pass
            return final_wav_path
        except Exception as e:
            logger.error(f"gTTS fallback TTS generation for '{lang_code_app}' failed: {e}", exc_info=True)
            return None

    def _generate_base_tts_audio(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        logger.debug(f"[_generate_base_tts_audio] Request for lang '{lang_code_app}', text: '{text[:70]}...'")
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map:
            polly_path = self._generate_base_tts_audio_with_polly(text, lang_code_app, temp_audio_dir)
            if polly_path and polly_path.exists(): return polly_path
            logger.warning(f"Polly TTS failed for '{lang_code_app}'. Attempting gTTS fallback...")
        else:
            if not BOTO3_AVAILABLE: logger.info(f"PollySDK (boto3) not available. Trying gTTS for '{lang_code_app}'.")
            else: logger.info(f"Polly voice not configured for '{lang_code_app}'. Trying gTTS.")
        
        gtts_path = self._generate_base_tts_audio_with_gtts(text, lang_code_app, temp_audio_dir)
        if gtts_path and gtts_path.exists(): return gtts_path
        
        logger.error(f"All TTS generation methods failed for lang '{lang_code_app}'.")
        return None

    def _clone_voice_with_api(self, ref_voice_audio_path: str, target_content_audio_path: str, output_cloned_audio_path: str, req_id_short: str = "clone") -> bool:
        logger.info(f"[{req_id_short}] _clone_voice_with_api: Ref='{Path(ref_voice_audio_path).name}', Content='{Path(target_content_audio_path).name}'")
        if not check_openvoice_api():
            logger.warning(f"[{req_id_short}] OpenVoice API NOT available at time of cloning. Skipping cloning.")
            return False
        if not Path(ref_voice_audio_path).exists() or Path(ref_voice_audio_path).stat().st_size < 1000:
            logger.error(f"[{req_id_short}] Reference voice audio MISSING/small: {ref_voice_audio_path}")
            return False
        if not Path(target_content_audio_path).exists() or Path(target_content_audio_path).stat().st_size < 1000:
            logger.error(f"[{req_id_short}] Target content audio (TTS) MISSING/small: {target_content_audio_path}")
            return False
        try:
            with open(ref_voice_audio_path, "rb") as f_ref, open(target_content_audio_path, "rb") as f_target_content:
                files_payload = {
                    "reference_audio_file": (Path(ref_voice_audio_path).name, f_ref, "audio/wav"),
                    "content_audio_file": (Path(target_content_audio_path).name, f_target_content, "audio/wav")
                }
                logger.debug(f"[{req_id_short}] Sending files to OpenVoice API /clone-voice. Keys: {list(files_payload.keys())}")
                api_clone_url = f"{OPENVOICE_API_URL}/clone-voice"
                response = requests.post(api_clone_url, files=files_payload, timeout=180)
            logger.info(f"[{req_id_short}] OpenVoice API /clone-voice response status: {response.status_code}")
            if response.status_code == 200:
                with open(output_cloned_audio_path, "wb") as f_out: f_out.write(response.content)
                if Path(output_cloned_audio_path).exists() and Path(output_cloned_audio_path).stat().st_size > 1000:
                    logger.info(f"[{req_id_short}] OpenVoice cloning successful. Output: {output_cloned_audio_path}")
                    return True
                else:
                    size_info = Path(output_cloned_audio_path).stat().st_size if Path(output_cloned_audio_path).exists() else "DOES NOT EXIST"
                    logger.error(f"[{req_id_short}] OpenVoice API OK, but output file problematic: {output_cloned_audio_path} (Size: {size_info})")
                    return False
            else:
                error_detail = f"OpenVoice API Error {response.status_code}"
                try: error_data = response.json(); error_detail += f" - Detail: {error_data.get('detail', response.text[:200])}"
                except json.JSONDecodeError: error_detail += f" - Response: {response.text[:200]}" if response.text else " (No error text)"
                logger.error(f"[{req_id_short}] {error_detail}")
                return False
        except requests.exceptions.RequestException as e_req:
            logger.error(f"[{req_id_short}] RequestException OpenVoice API call: {e_req}", exc_info=True); return False
        except Exception as e:
            logger.error(f"[{req_id_short}] General exception _clone_voice_with_api: {e}", exc_info=True); return False

    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str = "eng", target_lang: str = "fra") -> Dict[str, Any]:
        process_id_short = str(time.time_ns())[-6:]
        logger.info(f"[{process_id_short}] CascadedBackend.translate_speech CALLED. App Source: '{source_lang}', App Target: '{target_lang}'")

        if not self.initialized:
            logger.info(f"[{process_id_short}] Backend not initialized, attempting to initialize now...")
            try: self.initialize()
            except Exception as e_init:
                 logger.error(f"[{process_id_short}] CRITICAL: Backend failed to initialize during translate_speech: {e_init}", exc_info=True)
                 return {"audio": torch.zeros((1, 16000), dtype=torch.float32), "transcripts": {"source": "ASR_FAILED_BACKEND_INIT_ERROR", "target": "TRANSLATION_FAILED_BACKEND_INIT_ERROR"}}
            if not self.initialized:
                 logger.error(f"[{process_id_short}] CRITICAL: Backend STILL not initialized after attempt.")
                 return {"audio": torch.zeros((1, 16000), dtype=torch.float32), "transcripts": {"source": "ASR_FAILED_BACKEND_NOT_INIT", "target": "TRANSLATION_FAILED_BACKEND_NOT_INIT"}}

        start_time_translate_speech = time.time()
        with tempfile.TemporaryDirectory(prefix=f"cascaded_s2st_{process_id_short}_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            logger.info(f"[{process_id_short}] Using temp directory for processing: {temp_dir}")

            source_text, target_text = "ASR_FAILED_OR_SILENT", "TRANSLATION_FAILED_OR_NO_INPUT"
            output_tensor = torch.zeros((1, 16000), dtype=torch.float32) # Default silent audio

            try:
                asr_start_time = time.time()
                audio_numpy = audio_tensor.squeeze().cpu().numpy().astype(np.float32)

                if np.abs(audio_numpy).max() < 1e-5: # Near silence check
                    source_text = ""
                    logger.info(f"[{process_id_short}] Input audio near silent. ASR skipped.")
                elif WHISPER_AVAILABLE_FLAG and self.asr_model:
                    lang_hint = self._get_simple_lang_code(source_lang)
                    asr_result = self.asr_model.transcribe(audio_numpy, language=lang_hint if lang_hint != 'auto' else None, task="transcribe", fp16=(self.device.type == 'cuda'))
                    source_text = asr_result["text"]
                    logger.info(f"[{process_id_short}] Whisper ASR (Lang: {asr_result.get('language','unk')}): '{source_text[:70]}...' ({(time.time()-asr_start_time):.2f}s)")
                elif self.asr_pipeline:
                    temp_asr_path = temp_dir / f"asr_in_{process_id_short}.wav"; sf.write(str(temp_asr_path), audio_numpy, 16000)
                    source_text = self.asr_pipeline(str(temp_asr_path))["text"]
                    logger.info(f"[{process_id_short}] Transformers ASR: '{source_text[:70]}...' ({(time.time()-asr_start_time):.2f}s)")
                    try: os.remove(temp_asr_path)
                    except OSError: pass
                else:
                    logger.error(f"[{process_id_short}] No ASR model available. Cannot perform ASR.")
                    source_text = "ASR_MODEL_UNAVAILABLE"

                translation_start_time = time.time()
                if not source_text.strip() or source_text == "ASR_MODEL_UNAVAILABLE":
                    target_text = "" if not source_text.strip() else "TRANSLATION_SKIPPED_NO_ASR_TEXT"
                    logger.info(f"[{process_id_short}] Source text empty or ASR failed. Translation skipped.")
                else:
                    src_nllb_code, tgt_nllb_code = self._convert_to_nllb_code(source_lang), self._convert_to_nllb_code(target_lang)
                    logger.info(f"[{process_id_short}] Translating NLLB '{src_nllb_code}' to '{tgt_nllb_code}'.")
                    self.translator_tokenizer.src_lang = src_nllb_code
                    input_ids = self.translator_tokenizer(source_text, return_tensors="pt", padding=True).input_ids.to(self.device)
                    
                    forced_bos_token_id = None
                    try:
                        if hasattr(self.translator_tokenizer, 'get_lang_id'): forced_bos_token_id = self.translator_tokenizer.get_lang_id(tgt_nllb_code)
                        elif hasattr(self.translator_tokenizer, 'convert_tokens_to_ids'):
                            forced_bos_token_id = self.translator_tokenizer.convert_tokens_to_ids(tgt_nllb_code)
                            if forced_bos_token_id == self.translator_tokenizer.unk_token_id:
                                logger.warning(f"[{process_id_short}] NLLB token for '{tgt_nllb_code}' UNK. Trying alts.");
                                for token_try in [tgt_nllb_code, f"__{tgt_nllb_code}__", f"[{tgt_nllb_code}]", f"{tgt_nllb_code.split('_')[0]}_code"]:
                                    alt_bos_id = self.translator_tokenizer.convert_tokens_to_ids(token_try)
                                    if alt_bos_id != self.translator_tokenizer.unk_token_id: forced_bos_token_id = alt_bos_id; logger.info(f"[{process_id_short}] Found NLLB BOS for '{tgt_nllb_code}' as '{token_try}': {forced_bos_token_id}"); break
                                if forced_bos_token_id == self.translator_tokenizer.unk_token_id: logger.error(f"[{process_id_short}] NLLB BOS for '{tgt_nllb_code}' still UNK."); forced_bos_token_id = None
                        else: logger.error(f"[{process_id_short}] NLLB tokenizer no lang ID method."); forced_bos_token_id = None
                    except Exception as e_bos: logger.error(f"[{process_id_short}] Error NLLB BOS for {tgt_nllb_code}: {e_bos}", exc_info=True); forced_bos_token_id = None
                    
                    if forced_bos_token_id: logger.info(f"[{process_id_short}] Using BOS ID: {forced_bos_token_id} for {tgt_nllb_code}")
                    else: logger.warning(f"[{process_id_short}] No valid BOS for {tgt_nllb_code}. NLLB might default.")
                    
                    gen_kwargs = {"input_ids": input_ids, "max_length": max(256, len(input_ids[0]) * 3 + 50), "num_beams": 5, "length_penalty": 1.0, "early_stopping": True }
                    if forced_bos_token_id is not None: gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
                    with torch.no_grad(): translated_tokens = self.translator_model.generate(**gen_kwargs)
                    target_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                logger.info(f"[{process_id_short}] NLLB Translation: '{target_text[:70]}...' ({(time.time()-translation_start_time):.2f}s)")

                tts_start_time = time.time(); base_tts_audio_path = None
                if not target_text.strip() or target_text.startswith("TRANSLATION_"):
                    logger.warning(f"[{process_id_short}] Target text empty or indicates prior failure. TTS will produce silence.")
                else:
                    base_tts_audio_path = self._generate_base_tts_audio(target_text, target_lang, temp_dir)

                if base_tts_audio_path and base_tts_audio_path.exists():
                    y_base_tts, _ = librosa.load(str(base_tts_audio_path), sr=16000, mono=True)
                    output_tensor = torch.from_numpy(y_base_tts.astype(np.float32)).unsqueeze(0)
                    logger.info(f"[{process_id_short}] Base TTS: {base_tts_audio_path.name} ({(time.time()-tts_start_time):.2f}s)")
                else:
                    logger.error(f"[{process_id_short}] Base TTS failed. Output will be silence. ({(time.time()-tts_start_time):.2f}s)")

                current_ov_api_available = check_openvoice_api()
                # Conditions for cloning: config allows it, API is up, base TTS worked, and there was source text
                base_tts_is_ok = bool(base_tts_audio_path and base_tts_audio_path.exists())
                source_text_is_meaningful = bool(source_text.strip() and source_text not in ["ASR_FAILED_OR_SILENT", "ASR_MODEL_UNAVAILABLE"])
                should_attempt_cloning = self.use_voice_cloning_config and current_ov_api_available and base_tts_is_ok and source_text_is_meaningful

                logger.info(f"[{process_id_short}] Voice Cloning decision: UserConfig={self.use_voice_cloning_config}, OV_API_Up={current_ov_api_available}, BaseTTS_OK={base_tts_is_ok}, SourceText_Meaningful={source_text_is_meaningful} -> AttemptClone={should_attempt_cloning}")

                if should_attempt_cloning:
                    clone_t_start = time.time()
                    ref_path = temp_dir / f"orig_ref_{process_id_short}.wav"; sf.write(str(ref_path), audio_numpy, 16000)
                    cloned_path = temp_dir / f"cloned_final_{process_id_short}.wav"
                    clone_ok = self._clone_voice_with_api(str(ref_path), str(base_tts_audio_path), str(cloned_path), req_id_short=process_id_short)
                    if clone_ok and cloned_path.exists() and cloned_path.stat().st_size > 1000:
                        y_cloned, _ = librosa.load(str(cloned_path), sr=16000, mono=True)
                        output_tensor = torch.from_numpy(y_cloned.astype(np.float32)).unsqueeze(0)
                        logger.info(f"[{process_id_short}] Using OV CLONED audio. ({(time.time()-clone_t_start):.2f}s)")
                    else:
                        logger.warning(f"[{process_id_short}] OV cloning FAILED or output invalid. Using base TTS. ({(time.time()-clone_t_start):.2f}s)")
                else:
                    logger.info(f"[{process_id_short}] OV cloning SKIPPED based on conditions.")
                logger.info(f"[{process_id_short}] Final audio shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
            except Exception as e:
                logger.error(f"[{process_id_short}] Error in translate_speech pipeline: {e}", exc_info=True)
                output_tensor = torch.zeros((1, 16000),dtype=torch.float32) # Ensure silent output on error
                source_text = source_text if source_text not in ["ASR_FAILED_OR_SILENT", "ASR_MODEL_UNAVAILABLE"] else "PIPELINE_ASR_ERROR"
                target_text = "PIPELINE_TRANSLATION_OR_TTS_ERROR"

            logger.info(f"[{process_id_short}] translate_speech completed in {time.time() - start_time_translate_speech:.2f}s")
            return {"audio": output_tensor, "transcripts": {"source": source_text, "target": target_text}}

    def is_language_supported(self, lang_code_app: str) -> bool:
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map: return True
        if lang_code_app in self.simple_lang_code_map: return True # For gTTS fallback
        logger.warning(f"[is_language_supported] '{lang_code_app}' not supported by Polly or gTTS map.")
        return False

    def get_supported_languages(self) -> Dict[str, str]:
        supported = {}
        if BOTO3_AVAILABLE:
            for code, conf in self.polly_voice_map.items():
                supported[code] = self.display_language_names.get(code, f"{code.upper()} (Polly: {conf['VoiceId']})")
        for code_gtts, display_simple in self.simple_lang_code_map.items():
            if code_gtts not in supported: # Only add if not already covered by Polly
                display_name = self.display_language_names.get(code_gtts, f"{code_gtts.upper()} (gTTS Fallback: {display_simple})")
                if not (BOTO3_AVAILABLE and code_gtts in self.polly_voice_map): # Explicitly mark as fallback if not primary
                     display_name = self.display_language_names.get(code_gtts, code_gtts.upper()) + " (gTTS Fallback)"
                supported[code_gtts] = display_name
        if not supported:
            logger.warning("[get_supported_languages] No Polly or gTTS languages configured. Returning minimal fallback.")
            return {"eng": "English (Config Error)"}
        logger.debug(f"[get_supported_languages] Returning: {supported}")
        return supported

    def cleanup(self):
        logger.info("Cleaning CascadedBackend resources...")
        if hasattr(self, 'asr_model') and self.asr_model: del self.asr_model; self.asr_model = None; logger.debug("Whisper ASR model unloaded.")
        if hasattr(self, 'asr_pipeline') and self.asr_pipeline: del self.asr_pipeline; self.asr_pipeline = None; logger.debug("Transformers ASR pipeline unloaded.")
        if hasattr(self, 'translator_model') and self.translator_model: del self.translator_model; self.translator_model = None; logger.debug("NLLB Translator model unloaded.")
        if hasattr(self, 'translator_tokenizer') and self.translator_tokenizer: del self.translator_tokenizer; self.translator_tokenizer = None; logger.debug("NLLB Tokenizer unloaded.")
        if hasattr(self, 'polly_clients_cache'): self.polly_clients_cache.clear(); logger.debug("Polly clients cache cleared.")
        if self.device.type == 'cuda' and torch.cuda.is_available(): torch.cuda.empty_cache(); logger.debug("CUDA cache emptied.")
        import gc; gc.collect()
        logger.info("CascadedBackend resources cleaned.")