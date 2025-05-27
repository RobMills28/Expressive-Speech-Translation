# services/cascaded_backend.py
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List # Added List
import tempfile
import soundfile as sf
import librosa
import time
import traceback
from pydub import AudioSegment
import requests
import json
import sys
import uuid # For unique IDs
import re # For stripping SSML for gTTS

# --- AWS Polly (boto3) Import with DETAILED logging ---
BOTO3_AVAILABLE = False
boto3 = None
polly_init_logger = logging.getLogger(f"{__name__}.boto3_init_check")

try:
    polly_init_logger.info("Attempting to import 'boto3' for AWS Polly SDK...")
    import boto3 # type: ignore
    polly_init_logger.info(f"Successfully imported 'boto3'. Version: {getattr(boto3, '__version__', 'unknown')}")
    aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    polly_init_logger.info(f"Attempting to create a test Polly client in region: '{aws_default_region}'...")
    test_client = boto3.client('polly', region_name=aws_default_region)
    polly_init_logger.info("Testing Polly client with 'describe_voices'...")
    test_client.describe_voices(LanguageCode='en-US')
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
    if 'boto3' not in sys.modules: boto3 = None
# --- End AWS Polly Import ---

logger = logging.getLogger(__name__)

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
    global OPENVOICE_API_AVAILABLE
    try:
        status_url = f"{OPENVOICE_API_URL}/status"
        logger.debug(f"Checking OpenVoice API status at: {status_url}")
        response = requests.get(status_url, timeout=3)
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("tone_converter_model_loaded") and data.get("default_source_se_loaded"):
                    logger.info(f"OpenVoice API available and models loaded: {data.get('message', 'Status OK')}")
                    OPENVOICE_API_AVAILABLE = True; return True
                else:
                    logger.warning(f"OpenVoice API is UP ({response.status_code}), but models are NOT fully loaded: {data}")
                    OPENVOICE_API_AVAILABLE = False; return False
            except json.JSONDecodeError:
                logger.warning(f"OpenVoice API /status returned 200 but response was not valid JSON: {response.text[:200]}")
                OPENVOICE_API_AVAILABLE = False; return False
        elif response.status_code == 503:
            logger.warning(f"OpenVoice API /status returned 503: Service unavailable. Detail: {response.text[:200]}")
            OPENVOICE_API_AVAILABLE = False; return False
        else:
            logger.warning(f"OpenVoice API status check failed with HTTP {response.status_code}. Detail: {response.text[:200]}")
            OPENVOICE_API_AVAILABLE = False; return False
    except requests.exceptions.RequestException as e_req:
        logger.warning(f"OpenVoice API unavailable (RequestException): {type(e_req).__name__} - {e_req}")
        OPENVOICE_API_AVAILABLE = False; return False
    except Exception as e_gen:
        logger.warning(f"OpenVoice API status check encountered an unexpected error: {type(e_gen).__name__} - {e_gen}")
        OPENVOICE_API_AVAILABLE = False; return False

check_openvoice_api()

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

    def _get_polly_client(self, region_name: str) -> Optional[boto3.client]: # type: ignore
        if not BOTO3_AVAILABLE or not boto3:
            logger.warning("Boto3 SDK not available or not imported, cannot create Polly client.")
            return None
        if region_name in self.polly_clients_cache:
            logger.debug(f"Returning cached Polly client for region: {region_name}")
            return self.polly_clients_cache[region_name]
        try:
            logger.info(f"Creating new Polly client for region: {region_name}")
            client = boto3.client('polly', region_name=region_name)
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
                                                device=0 if self.device.type == 'cuda' else -1)
                logger.info("Transformers ASR pipeline loaded ('openai/whisper-base').")

            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            if self.device.type == 'cuda' and torch.cuda.is_available():
                self.translator_model = self.translator_model.to(self.device)
            logger.info("NLLB translation model and tokenizer loaded.")

            if BOTO3_AVAILABLE:
                default_aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
                self._get_polly_client(default_aws_region)

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

    def _get_text_and_pauses_from_asr(self, audio_numpy: np.ndarray, source_lang: str, process_id_short: str) -> tuple[str, List[Dict[str, Any]]]:
        text_segments = []
        pauses_info: List[Dict[str, Any]] = []
        last_word_end_time = 0.0
        word_count = -1 # word_count will be the index of the word *before* which a pause might be inserted

        if np.abs(audio_numpy).max() < 1e-5:
            logger.info(f"[{process_id_short}] Input audio near silent for ASR.")
            return "", []

        if WHISPER_AVAILABLE_FLAG and self.asr_model:
            lang_hint = self._get_simple_lang_code(source_lang)
            logger.info(f"[{process_id_short}] Running Whisper ASR with word_timestamps=True, lang_hint='{lang_hint}'")
            asr_result = self.asr_model.transcribe(
                audio_numpy,
                language=lang_hint if lang_hint != 'auto' else None,
                task="transcribe",
                fp16=(self.device.type == 'cuda'),
                word_timestamps=True,
            )
            # For detailed debugging of Whisper's output structure:
            # logger.debug(f"[{process_id_short}] Full Whisper ASR result with timestamps: {json.dumps(asr_result, indent=2)}")

            if "segments" in asr_result:
                current_global_word_index = -1
                for segment_idx, segment in enumerate(asr_result["segments"]):
                    segment_word_list = [] # Collect words for this segment first
                    if "words" in segment and segment["words"]:
                        for word_info_idx, word_info_item in enumerate(segment["words"]):
                            current_global_word_index += 1
                            # Handle both dict and potential WhisperWord object from different Whisper versions/usages
                            if isinstance(word_info_item, dict):
                                word_text = word_info_item.get("word", "").strip()
                                start_time = word_info_item.get("start", 0.0)
                                end_time = word_info_item.get("end", 0.0)
                            elif hasattr(word_info_item, 'word') and hasattr(word_info_item, 'start') and hasattr(word_info_item, 'end'):
                                word_text = word_info_item.word.strip()
                                start_time = word_info_item.start
                                end_time = word_info_item.end
                            else:
                                logger.warning(f"[{process_id_short}] Segment {segment_idx}, Word {word_info_idx}: Unexpected word_info format: {word_info_item}")
                                continue
                            
                            # Detect pause *before* the current word if it's not the very first word overall
                            if current_global_word_index > 0 and start_time > last_word_end_time: # last_word_end_time is from the previous word
                                pause_duration = start_time - last_word_end_time
                                if pause_duration > 0.250: # 250ms threshold
                                    pauses_info.append({
                                        "start": round(last_word_end_time, 3), # Pause starts after last word ends
                                        "end": round(start_time, 3),       # Pause ends before current word starts
                                        "duration": round(pause_duration, 3),
                                        # Pause occurs *after* the (current_global_word_index - 1)-th word
                                        "insert_after_word_index": current_global_word_index - 1
                                    })
                                    logger.debug(f"[{process_id_short}] Detected pause: {pause_duration:.3f}s after word index {current_global_word_index - 1} ('{text_segments[-1] if text_segments else '<START>'}')")
                            
                            segment_word_list.append(word_text)
                            last_word_end_time = end_time # Update for the next word
                        
                        if segment_word_list: # Add collected words from this segment
                            text_segments.extend(segment_word_list)

                    elif "text" in segment: # Fallback if no 'words' but 'text' in segment
                         segment_text = segment.get("text", "").strip()
                         if segment_text:
                            # If we have segment text but no words, we lose precise word count for pauses
                            # Split by space as a rough estimate for word count increment
                            num_words_in_segment = len(segment_text.split())
                            current_global_word_index += num_words_in_segment
                            text_segments.append(segment_text)
                            logger.warning(f"[{process_id_short}] Segment {segment_idx} has text but no word timestamps. Pause detection for this segment might be inaccurate.")
                            last_word_end_time = segment.get("end", last_word_end_time) # Update with segment end if available


            source_text = " ".join(text_segments).strip()
            # If joining segments still yields empty, but Whisper had a top-level "text"
            if not source_text and asr_result.get("text"):
                source_text = asr_result.get("text", "").strip()
                logger.warning(f"[{process_id_short}] Word/segment processing resulted in empty text, using full ASR text. Pause info will be empty.")
                pauses_info = [] # Reset pauses as their indices would be invalid

            logger.info(f"[{process_id_short}] Whisper ASR (Language: {asr_result.get('language','unk')}): '{source_text[:70]}...'")
            logger.info(f"[{process_id_short}] Detected {len(pauses_info)} significant pauses.")
            if pauses_info: logger.debug(f"[{process_id_short}] Pauses info: {json.dumps(pauses_info, indent=2)}")


        elif self.asr_pipeline:
            temp_asr_path = Path(tempfile.gettempdir()) / f"asr_in_{process_id_short}.wav"
            sf.write(str(temp_asr_path), audio_numpy, 16000)
            source_text = self.asr_pipeline(str(temp_asr_path))["text"]
            logger.info(f"[{process_id_short}] Transformers ASR: '{source_text[:70]}...' (Pause detection not implemented for this ASR path)")
            try: os.remove(temp_asr_path)
            except OSError: pass
        else:
            logger.error(f"[{process_id_short}] No ASR model available.")
            return "ASR_MODEL_UNAVAILABLE", []

        return source_text, pauses_info

    def _insert_pauses_into_text(self, text: str, original_source_pauses: List[Dict[str, Any]], source_text_for_alignment: str) -> str:
        if not original_source_pauses or not text.strip():
            return text

        target_words = text.split()
        if not target_words: return text

        source_words_for_alignment = source_text_for_alignment.split()
        if not source_words_for_alignment:
            logger.warning("Source text for alignment (for pause insertion) is empty. Cannot insert pauses based on source.")
            return text

        modified_target_words = []
        ssml_inserted_flag = False
        
        # Sort pauses by their insertion point to process them in order
        sorted_pauses = sorted(original_source_pauses, key=lambda p: p['insert_after_word_index'])
        
        current_pause_list_idx = 0
        
        for target_word_idx, target_word_val in enumerate(target_words):
            modified_target_words.append(target_word_val)
            
            # Heuristic: Map current target word index to an approximate source word index
            # This is a very rough alignment.
            # The idea is to see if the *source* word index that this target word *might* correspond to
            # is one after which a pause should be inserted.
            approx_source_word_idx_this_target_word_represents = round(
                ((target_word_idx + 1) / len(target_words)) * len(source_words_for_alignment)
            ) -1 # Convert to 0-based index

            # Check if any pending pause should be inserted *after* the source word this target word represents
            if current_pause_list_idx < len(sorted_pauses):
                pause_info = sorted_pauses[current_pause_list_idx]
                # If the source word index for the current pause is less than or equal to
                # the approximated source index our current target word represents,
                # it means this pause should have occurred by now (or right now).
                if pause_info['insert_after_word_index'] <= approx_source_word_idx_this_target_word_represents:
                    pause_duration_ms = int(pause_info['duration'] * 1000)
                    if pause_duration_ms >= 100: # Minimum 100ms for a meaningful SSML break
                        ssml_break_tag = f'<break time="{pause_duration_ms}ms"/>'
                        modified_target_words.append(ssml_break_tag)
                        ssml_inserted_flag = True
                        logger.debug(f"Inserted SSML pause: {ssml_break_tag} after target word '{target_word_val}' (orig source word_idx {pause_info['insert_after_word_index']})")
                    current_pause_list_idx += 1 # Move to the next pause in the sorted list
        
        result_text = " ".join(modified_target_words)

        if ssml_inserted_flag:
            return f"<speak>{result_text}</speak>"
        else:
            return text # Return original text if no pauses were inserted

    def _generate_base_tts_audio_with_polly(self, text: str, lang_code_app: str, temp_audio_dir: Path, use_ssml: bool = False) -> Optional[Path]:
        polly_voice_config = self.polly_voice_map.get(lang_code_app)
        if not polly_voice_config:
            logger.warning(f"No Polly voice configuration for app lang code '{lang_code_app}'.")
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
                # Ensure text is wrapped in <speak> if it's SSML and doesn't already have it
                if not text.strip().lower().startswith("<speak>"):
                     logger.warning(f"SSML flag is true for Polly, but text input does not start with <speak>. Text: '{text[:100]}...' Will wrap it.")
                     req_params['Text'] = f"<speak>{text}</speak>"
            
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
        except boto3.exceptions.Boto3Error as e_boto: # type: ignore
             logger.error(f"Boto3/Polly API Error for '{lang_code_app}': {e_boto}", exc_info=False)
             logger.error(f"Details: Type={type(e_boto).__name__}, Args={e_boto.args}")
             if hasattr(e_boto, 'response') and e_boto.response and 'Error' in e_boto.response: # type: ignore
                 logger.error(f"AWS Error Code: {e_boto.response['Error'].get('Code')}, Message: {e_boto.response['Error'].get('Message')}") # type: ignore
             return None
        except Exception as e:
            logger.error(f"Generic Polly TTS generation failure for '{lang_code_app}': {e}", exc_info=True)
            return None

    def _generate_base_tts_audio_with_gtts(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        text_for_gtts = text
        if "<speak>" in text.lower() and "</speak>" in text.lower(): # Basic check for SSML-like structure
            # Attempt to strip <speak> tags and <break .../> tags for gTTS
            text_for_gtts = re.sub(r'(?i)<\s*speak\s*>', '', text_for_gtts)
            text_for_gtts = re.sub(r'(?i)<\s*/\s*speak\s*>', '', text_for_gtts)
            text_for_gtts = re.sub(r'(?i)<\s*break\s+time\s*=\s*"[^"]*"\s*/?\s*>', ' ', text_for_gtts) # Replace break with a space
            text_for_gtts = re.sub(r'\s+', ' ', text_for_gtts).strip() # Normalize spaces
            if text_for_gtts != text:
                 logger.warning(f"SSML-like tags stripped for gTTS. Original text sample: '{text[:70]}...', Stripped text sample: '{text_for_gtts[:70]}...'")

        logger.warning(f"Using Fallback gTTS for language '{lang_code_app}'. Text (after potential SSML strip): '{text_for_gtts[:70]}...'")
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
            tts = gTTS(text=text_for_gtts, lang=gtts_lang_code, slow=False)
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

    def _generate_base_tts_audio(self, text: str, lang_code_app: str, temp_audio_dir: Path, use_ssml: bool = False) -> Optional[Path]:
        logger.debug(f"[_generate_base_tts_audio] Request for lang '{lang_code_app}', SSML: {use_ssml}, text: '{text[:70]}...'")
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map:
            polly_path = self._generate_base_tts_audio_with_polly(text, lang_code_app, temp_audio_dir, use_ssml=use_ssml)
            if polly_path and polly_path.exists(): return polly_path
            logger.warning(f"Polly TTS failed for '{lang_code_app}' (SSML={use_ssml}). Attempting gTTS fallback...")
        else:
            if not BOTO3_AVAILABLE: logger.info(f"PollySDK (boto3) not available. Trying gTTS for '{lang_code_app}'.")
            else: logger.info(f"Polly voice not configured for '{lang_code_app}'. Trying gTTS.")
        
        gtts_path = self._generate_base_tts_audio_with_gtts(text, lang_code_app, temp_audio_dir) # gTTS stripping logic is internal to this function now
        if gtts_path and gtts_path.exists(): return gtts_path
        
        logger.error(f"All TTS generation methods failed for lang '{lang_code_app}'.")
        return None

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

            source_text_raw, source_pauses = "ASR_FAILED_OR_SILENT", []
            target_text_raw = "TRANSLATION_FAILED_OR_NO_INPUT"
            output_tensor = torch.zeros((1, 16000), dtype=torch.float32)

            try:
                asr_start_time = time.time()
                audio_numpy = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
                source_text_raw, source_pauses = self._get_text_and_pauses_from_asr(audio_numpy, source_lang, process_id_short)
                logger.info(f"[{process_id_short}] ASR processing time: {(time.time()-asr_start_time):.2f}s. Source Text: '{source_text_raw[:100]}...'. Found {len(source_pauses)} pauses.")

                translation_start_time = time.time()
                if not source_text_raw.strip() or source_text_raw == "ASR_MODEL_UNAVAILABLE":
                    target_text_raw = "" if not source_text_raw.strip() else "TRANSLATION_SKIPPED_NO_ASR_TEXT"
                    logger.info(f"[{process_id_short}] Source text empty or ASR failed. Translation skipped.")
                else:
                    src_nllb_code, tgt_nllb_code = self._convert_to_nllb_code(source_lang), self._convert_to_nllb_code(target_lang)
                    logger.info(f"[{process_id_short}] Translating NLLB '{src_nllb_code}' to '{tgt_nllb_code}'.")
                    self.translator_tokenizer.src_lang = src_nllb_code
                    input_ids = self.translator_tokenizer(source_text_raw, return_tensors="pt", padding=True).input_ids.to(self.device)
                    
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
                    target_text_raw = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                logger.info(f"[{process_id_short}] NLLB Raw Translation: '{target_text_raw[:70]}...' ({(time.time()-translation_start_time):.2f}s)")

                target_text_for_tts = self._insert_pauses_into_text(target_text_raw, source_pauses, source_text_raw)
                is_ssml_text = target_text_for_tts.strip().lower().startswith("<speak>")
                logger.info(f"[{process_id_short}] Text for TTS (SSML={is_ssml_text}): '{target_text_for_tts[:100]}...'")

                tts_start_time = time.time(); base_tts_audio_path = None
                if not target_text_raw.strip() or target_text_raw.startswith("TRANSLATION_"):
                    logger.warning(f"[{process_id_short}] Target text empty or indicates prior failure. TTS will produce silence.")
                else:
                    base_tts_audio_path = self._generate_base_tts_audio(target_text_for_tts, target_lang, temp_dir, use_ssml=is_ssml_text)

                if base_tts_audio_path and base_tts_audio_path.exists():
                    y_base_tts, _ = librosa.load(str(base_tts_audio_path), sr=16000, mono=True)
                    output_tensor = torch.from_numpy(y_base_tts.astype(np.float32)).unsqueeze(0)
                    logger.info(f"[{process_id_short}] Base TTS: {base_tts_audio_path.name} ({(time.time()-tts_start_time):.2f}s)")
                else:
                    logger.error(f"[{process_id_short}] Base TTS failed. Output will be silence. ({(time.time()-tts_start_time):.2f}s)")

                current_ov_api_available = check_openvoice_api()
                base_tts_is_ok = bool(base_tts_audio_path and base_tts_audio_path.exists())
                source_text_is_meaningful = bool(source_text_raw.strip() and source_text_raw not in ["ASR_FAILED_OR_SILENT", "ASR_MODEL_UNAVAILABLE"])
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
                output_tensor = torch.zeros((1, 16000),dtype=torch.float32)
                source_text_raw = source_text_raw if source_text_raw not in ["ASR_FAILED_OR_SILENT", "ASR_MODEL_UNAVAILABLE"] else "PIPELINE_ASR_ERROR"
                target_text_raw = "PIPELINE_TRANSLATION_OR_TTS_ERROR"

            logger.info(f"[{process_id_short}] translate_speech completed in {time.time() - start_time_translate_speech:.2f}s")
            return {"audio": output_tensor, "transcripts": {"source": source_text_raw, "target": target_text_raw}}

    def _clone_voice_with_api(self, ref_voice_audio_path: str, target_content_audio_path: str, output_cloned_audio_path: str, req_id_short: str = "clone") -> bool:
        logger.info(f"[{req_id_short}] _clone_voice_with_api: Ref='{Path(ref_voice_audio_path).name}', Content='{Path(target_content_audio_path).name}'")
        if not check_openvoice_api():
            logger.warning(f"[{req_id_short}] OpenVoice API NOT available at time of cloning. Skipping cloning.")
            return False
        if not Path(ref_voice_audio_path).exists() or Path(ref_voice_audio_path).stat().st_size < 1000:
            logger.error(f"[{req_id_short}] Reference voice audio MISSING/small: {ref_voice_audio_path}"); return False
        if not Path(target_content_audio_path).exists() or Path(target_content_audio_path).stat().st_size < 1000:
            logger.error(f"[{req_id_short}] Target content audio (TTS) MISSING/small: {target_content_audio_path}"); return False
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

    def is_language_supported(self, lang_code_app: str) -> bool:
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map: return True
        if lang_code_app in self.simple_lang_code_map: return True
        logger.warning(f"[is_language_supported] '{lang_code_app}' not supported by Polly or gTTS map.")
        return False

    def get_supported_languages(self) -> Dict[str, str]:
        supported = {}
        if BOTO3_AVAILABLE:
            for code, conf in self.polly_voice_map.items():
                supported[code] = self.display_language_names.get(code, f"{code.upper()} (Polly: {conf['VoiceId']})")
        for code_gtts, display_simple in self.simple_lang_code_map.items():
            if code_gtts not in supported:
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
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache(); logger.debug("CUDA cache emptied.")
        import gc; gc.collect()
        logger.info("CascadedBackend resources cleaned.")