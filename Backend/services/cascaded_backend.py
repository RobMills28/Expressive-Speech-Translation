# services/cascaded_backend.py
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import soundfile as sf
import librosa
import time
import traceback
from pydub import AudioSegment
import requests
import json
import sys
import uuid
import re

# Attempt to import noisereduce
try:
    import noisereduce
    NOISEREDUCE_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported 'noisereduce' library.")
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logging.getLogger(__name__).warning("'noisereduce' library not found. Noise reduction for reference audio will be skipped.")

# Attempt to import webrtcvad for more precise VAD
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported 'webrtcvad' library for VAD.")
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logging.getLogger(__name__).warning("'webrtcvad' library not found. Using basic silence removal for reference audio VAD.")

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
    BOTO3_AVAILABLE = False; boto3 = None
except Exception as e_client_or_general:
    polly_init_logger.error(f"BOTO3 CLIENT FAILURE: 'boto3' imported, but Polly client creation/test FAILED: {type(e_client_or_general).__name__} - {e_client_or_general}", exc_info=False)
    polly_init_logger.error("DETAILS: This usually means AWS credentials or region are not correctly configured, or network/IAM permission issues.")
    BOTO3_AVAILABLE = False;
    if 'boto3' not in sys.modules: boto3 = None

logger = logging.getLogger(__name__)

try:
    import whisper # type: ignore
    WHISPER_AVAILABLE_FLAG = True
    logger.info("Whisper library FOUND and imported for CascadedBackend.")
except ImportError:
    logger.warning("Whisper library NOT FOUND. Falling back to Transformers pipeline.")
    WHISPER_AVAILABLE_FLAG = False; whisper = None

OPENVOICE_API_URL = os.getenv("OPENVOICE_API_URL", "http://localhost:8000") # For OpenVoice cloning service
VOICE_SIMILARITY_API_URL = os.getenv("VOICE_SIMILARITY_API_URL", "http://localhost:8001") # For the new similarity service
OPENVOICE_API_AVAILABLE = False

def check_openvoice_api():
    global OPENVOICE_API_AVAILABLE
    try:
        status_url = f"{OPENVOICE_API_URL}/status"
        logger.debug(f"Checking OpenVoice API status at: {status_url}")
        response = requests.get(status_url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("tone_converter_model_loaded") and data.get("default_source_se_loaded"):
                logger.info(f"OpenVoice API available and models loaded: {data.get('message', 'Status OK')}")
                OPENVOICE_API_AVAILABLE = True; return True
            else:
                logger.warning(f"OpenVoice API UP ({response.status_code}), but models NOT fully loaded: {data}")
                OPENVOICE_API_AVAILABLE = False; return False
        elif response.status_code == 503:
            logger.warning(f"OpenVoice API /status 503: Service unavailable. Detail: {response.text[:200]}")
            OPENVOICE_API_AVAILABLE = False; return False
        else:
            logger.warning(f"OpenVoice API status check failed HTTP {response.status_code}. Detail: {response.text[:200]}")
            OPENVOICE_API_AVAILABLE = False; return False
    except json.JSONDecodeError:
        logger.warning(f"OpenVoice API /status 200 but not valid JSON: {response.text[:200]}")
        OPENVOICE_API_AVAILABLE = False; return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"OpenVoice API unavailable (RequestException): {type(e).__name__} - {e}")
        OPENVOICE_API_AVAILABLE = False; return False
    except Exception as e_gen:
        logger.warning(f"OpenVoice API status check unexpected error: {type(e_gen).__name__} - {e_gen}")
        OPENVOICE_API_AVAILABLE = False; return False
    OPENVOICE_API_AVAILABLE = False; return False

check_openvoice_api()

from .translation_strategy import TranslationBackend
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline
# Removed: from .voice_similarity_analyzer import VoiceSimilarityAnalyzer

class CascadedBackend(TranslationBackend):
    def __init__(self, device=None, use_voice_cloning=True):
        logger.info(f"Initializing CascadedBackend. Device: {device}, Voice Cloning Config: {use_voice_cloning}, PollySDK: {BOTO3_AVAILABLE}, Whisper: {WHISPER_AVAILABLE_FLAG}, NoiseReduce: {NOISEREDUCE_AVAILABLE}, WebRTCVAD: {WEBRTCVAD_AVAILABLE}")
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"CascadedBackend ML device: {self.device}")
        self.initialized = False
        self.use_voice_cloning_config = use_voice_cloning
        self.polly_voice_map = {'eng':{'VoiceId':'Joanna','Engine':'neural','Region':'us-east-1'},'deu':{'VoiceId':'Vicki','Engine':'neural','Region':'eu-central-1'},'spa':{'VoiceId':'Lucia','Engine':'neural','Region':'eu-west-1'},'fra':{'VoiceId':'Lea','Engine':'neural','Region':'eu-west-1'},'ita':{'VoiceId':'Bianca','Engine':'neural','Region':'eu-central-1'},'jpn':{'VoiceId':'Kazuha','Engine':'neural','Region':'ap-northeast-1'},'kor':{'VoiceId':'Seoyeon','Engine':'neural','Region':'ap-northeast-2'},'por':{'VoiceId':'Ines','Engine':'neural','Region':'eu-west-1'},'rus':{'VoiceId':'Olga','Engine':'neural','Region':'eu-central-1'},'cmn':{'VoiceId':'Zhiyu','Engine':'neural','Region':'ap-northeast-1'},'ara':{'VoiceId':'Hala','Engine':'neural','Region':'me-south-1'},'hin':{'VoiceId':'Kajal','Engine':'neural','Region':'ap-south-1'},'nld':{'VoiceId':'Laura','Engine':'neural','Region':'eu-west-1'},'pol':{'VoiceId':'Ewa','Engine':'neural','Region':'eu-central-1'},'tur':{'VoiceId':'Burcu','Engine':'neural','Region':'eu-central-1'},'ukr':{'VoiceId':'Olga','Engine':'neural','Region':'eu-central-1'}}
        self.simple_lang_code_map = { 'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it', 'por': 'pt', 'cmn': 'zh-cn', 'jpn': 'ja', 'kor': 'ko', 'ara': 'ar', 'hin': 'hi', 'nld': 'nl', 'rus': 'ru', 'pol': 'pl', 'tur': 'tr', 'ukr': 'uk'}
        self.display_language_names = {'eng':'English (Joanna, Neural)','deu':'German (Vicki, Neural)','spa':'Spanish (Lucia, Neural)','fra':'French (Lea, Neural)','ita':'Italian (Bianca, Neural)','jpn':'Japanese (Kazuha, Neural)','kor':'Korean (Seoyeon, Neural)','por':'Portuguese (Ines, Eur., Neural)','rus':'Russian (Olga, Neural)','cmn':'Mandarin (Zhiyu, Neural)','ara':'Arabic (Hala, Egy., Neural)','hin':'Hindi (Kajal, Neural)','nld':'Dutch (Laura, Neural)','pol':'Polish (Ewa, Neural)','tur':'Turkish (Burcu, Neural)','ukr':'Ukrainian (Fallback Olga, Rus., Neural)'}
        self.asr_model = None; self.asr_pipeline = None; self.translator_model = None; self.translator_tokenizer = None; self.polly_clients_cache = {}
        self.vad_mode = 3
        if WEBRTCVAD_AVAILABLE:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(self.vad_mode)
        else:
            self.vad = None
        logger.info(f"CascadedBackend __init__ final flags: Whisper={WHISPER_AVAILABLE_FLAG}, OpenVoiceAPI={OPENVOICE_API_AVAILABLE}, PollySDK={BOTO3_AVAILABLE}")

    def _get_polly_client(self, region_name: str) -> Optional[boto3.client]: # type: ignore
        if not BOTO3_AVAILABLE or not boto3:
            logger.warning("Boto3 SDK not available. Cannot create Polly client.")
            return None
        if region_name in self.polly_clients_cache:
            return self.polly_clients_cache[region_name]
        try:
            logger.info(f"Creating Polly client for region: {region_name}")
            client = boto3.client('polly', region_name=region_name)
            self.polly_clients_cache[region_name] = client
            return client
        except Exception as e:
            logger.error(f"Failed to create Polly client for {region_name}: {e}", exc_info=True)
            return None

    def initialize(self):
        if self.initialized: return
        logger.info(f"Initializing CascadedBackend components on device: {self.device}")
        start_time = time.time()
        try:
            if WHISPER_AVAILABLE_FLAG and whisper:
                self.asr_model = whisper.load_model("medium", device=self.device)
                logger.info("Whisper ASR loaded ('medium').")
            else:
                self.asr_pipeline = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if self.device.type == 'cuda' else -1)
                logger.info("Transformers ASR pipeline loaded ('openai/whisper-base').")
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(self.device)
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            logger.info("NLLB translation model and tokenizer loaded.")
            if BOTO3_AVAILABLE: self._get_polly_client(os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            # No need to preload VoiceSimilarityAnalyzer model here anymore
            self.initialized = True
            logger.info(f"CascadedBackend initialized successfully in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize CascadedBackend: {e}", exc_info=True)
            self.initialized = False; raise

    def _convert_to_nllb_code(self, lang_code_app: str) -> str:
        mapping = {'eng':'eng_Latn','fra':'fra_Latn','spa':'spa_Latn','deu':'deu_Latn','ita':'ita_Latn','por':'por_Latn','rus':'rus_Cyrl','cmn':'zho_Hans','jpn':'jpn_Jpan','kor':'kor_Hang','ara':'ara_Arab','hin':'hin_Deva','nld':'nld_Latn','pol':'pol_Latn','tur':'tur_Latn','ukr':'ukr_Cyrl','ces':'ces_Latn','hun':'hun_Latn'}
        return mapping.get(lang_code_app.lower(), 'eng_Latn')

    def _get_simple_lang_code(self, lang_code_app: str) -> str:
        return self.simple_lang_code_map.get(lang_code_app.lower(), 'en')

    def _get_text_and_pauses_from_asr(self, audio_numpy: np.ndarray, source_lang: str, process_id_short: str) -> tuple[str, List[Dict[str, Any]]]:
        text_segments = []
        pauses_info: List[Dict[str, Any]] = []
        last_word_end_time = 0.0
        if np.abs(audio_numpy).max() < 1e-5:
            logger.info(f"[{process_id_short}] Input audio near silent for ASR.")
            return "", []
        if WHISPER_AVAILABLE_FLAG and self.asr_model:
            lang_hint = self._get_simple_lang_code(source_lang)
            logger.info(f"[{process_id_short}] Running Whisper ASR with word_timestamps=True, lang_hint='{lang_hint}'")
            asr_result = self.asr_model.transcribe(audio_numpy, language=lang_hint if lang_hint != 'auto' else None, task="transcribe", fp16=(self.device.type == 'cuda'), word_timestamps=True)
            if "segments" in asr_result:
                current_global_word_index = -1
                for segment_idx, segment in enumerate(asr_result["segments"]):
                    segment_word_list = []
                    if "words" in segment and segment["words"]:
                        for word_info_idx, word_info_item in enumerate(segment["words"]):
                            current_global_word_index += 1
                            if isinstance(word_info_item, dict):
                                word_text, start_time, end_time = word_info_item.get("word", "").strip(), word_info_item.get("start", 0.0), word_info_item.get("end", 0.0)
                            elif hasattr(word_info_item, 'word'):
                                word_text, start_time, end_time = word_info_item.word.strip(), word_info_item.start, word_info_item.end
                            else: continue
                            if current_global_word_index > 0 and start_time > last_word_end_time:
                                pause_duration = start_time - last_word_end_time
                                if pause_duration > 0.250:
                                    pauses_info.append({"start": round(last_word_end_time, 3), "end": round(start_time, 3), "duration": round(pause_duration, 3), "insert_after_word_index": current_global_word_index - 1})
                                    logger.debug(f"[{process_id_short}] Detected pause: {pause_duration:.3f}s after word idx {current_global_word_index - 1} ('{text_segments[-1] if text_segments else '<S>'}')")
                            segment_word_list.append(word_text)
                            last_word_end_time = end_time
                        if segment_word_list: text_segments.extend(segment_word_list)
                    elif "text" in segment and (segment_text := segment.get("text", "").strip()):
                        current_global_word_index += len(segment_text.split())
                        text_segments.append(segment_text)
                        logger.warning(f"[{process_id_short}] Segment {segment_idx} no word timestamps. Pause detection impacted.")
                        last_word_end_time = segment.get("end", last_word_end_time)
            source_text = " ".join(text_segments).strip()
            if not source_text and asr_result.get("text"):
                source_text = asr_result.get("text", "").strip(); pauses_info = []
                logger.warning(f"[{process_id_short}] Segments processing empty, using full ASR text. Pauses lost.")
            logger.info(f"[{process_id_short}] Whisper ASR (Lang: {asr_result.get('language','unk')}): '{source_text[:70]}...'")
            logger.info(f"[{process_id_short}] Detected {len(pauses_info)} sig. pauses.")
            if pauses_info: logger.debug(f"[{process_id_short}] Pauses: {json.dumps(pauses_info, indent=2)}")
        elif self.asr_pipeline:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=Path(tempfile.gettempdir())) as tmp_asr_file:
                sf.write(tmp_asr_file.name, audio_numpy, 16000)
                source_text = self.asr_pipeline(tmp_asr_file.name)["text"]
            try: os.unlink(tmp_asr_file.name) 
            except: pass
            logger.info(f"[{process_id_short}] Transformers ASR: '{source_text[:70]}...' (No pause detection)")
        else: return "ASR_MODEL_UNAVAILABLE", []
        return source_text, pauses_info

    def _insert_pauses_into_text(self, text: str, original_source_pauses: List[Dict[str, Any]], source_text_for_alignment: str) -> str:
        if not original_source_pauses or not text.strip(): return text
        target_words = text.split();
        if not target_words: return text
        source_words_for_alignment = source_text_for_alignment.split()
        if not source_words_for_alignment: logger.warning("Source text for alignment empty. Cannot insert pauses."); return text
        modified_target_words = []; ssml_inserted_flag = False
        sorted_pauses = sorted(original_source_pauses, key=lambda p: p['insert_after_word_index'])
        current_pause_list_idx = 0
        for target_word_idx, target_word_val in enumerate(target_words):
            modified_target_words.append(target_word_val)
            approx_source_word_idx = round(((target_word_idx + 0.5) / len(target_words)) * len(source_words_for_alignment)) -1
            if current_pause_list_idx < len(sorted_pauses):
                pause_info = sorted_pauses[current_pause_list_idx]
                if pause_info['insert_after_word_index'] <= approx_source_word_idx:
                    pause_duration_ms = int(pause_info['duration'] * 1000)
                    if pause_duration_ms >= 100:
                        ssml_break_tag = f'<break time="{pause_duration_ms}ms"/>'
                        modified_target_words.append(ssml_break_tag); ssml_inserted_flag = True
                        logger.debug(f"Inserted SSML pause: {ssml_break_tag} after target word '{target_word_val}' (orig src_idx {pause_info['insert_after_word_index']})")
                    current_pause_list_idx += 1
        return f"<speak>{' '.join(modified_target_words)}</speak>" if ssml_inserted_flag else text

    def _preprocess_reference_audio(self, audio_numpy: np.ndarray, sr: int, process_id_short: str) -> np.ndarray:
        logger.info(f"[{process_id_short}] Preprocessing reference audio for voice cloning (length: {len(audio_numpy)/sr:.2f}s)...")
        processed_audio = audio_numpy.copy()
        original_length_sec = len(processed_audio) / sr

        if WEBRTCVAD_AVAILABLE and self.vad:
            logger.debug(f"[{process_id_short}] Applying webrtcvad (mode set to: {self.vad_mode})...")
            frame_duration_ms = 30; samples_per_frame = int(sr * frame_duration_ms / 1000)
            speech_segments_data = []
            pcm_audio = (processed_audio * 32767).astype(np.int16)
            for i in range(0, len(pcm_audio) - samples_per_frame + 1, samples_per_frame):
                frame = pcm_audio[i:i+samples_per_frame].tobytes()
                if len(frame) == samples_per_frame * 2:
                    if self.vad.is_speech(frame, sr):
                        speech_segments_data.append(processed_audio[i:i+samples_per_frame])
            if speech_segments_data:
                processed_audio = np.concatenate(speech_segments_data)
                vad_length_sec = len(processed_audio)/sr
                logger.info(f"[{process_id_short}] VAD applied. Original: {original_length_sec:.2f}s, VAD output: {vad_length_sec:.2f}s")
                if vad_length_sec < 1.0:
                    logger.warning(f"[{process_id_short}] VAD output very short ({vad_length_sec:.2f}s). Reverting to original audio for reference.")
                    processed_audio = audio_numpy.copy()
            else:
                logger.warning(f"[{process_id_short}] VAD found no speech. Using original audio for reference.")
                processed_audio = audio_numpy.copy()
        else:
            logger.debug(f"[{process_id_short}] Using librosa basic silence removal (top_db=30).")
            non_silent_parts = librosa.effects.split(processed_audio, top_db=30)
            if non_silent_parts.size > 0 :
                processed_audio = np.concatenate([processed_audio[start:end] for start, end in non_silent_parts])
                logger.info(f"[{process_id_short}] Basic silence removal applied. Original: {original_length_sec:.2f}s, Output: {len(processed_audio)/sr:.2f}s")
            else: logger.warning(f"[{process_id_short}] Basic silence removal found no speech. Using original audio.")

        if NOISEREDUCE_AVAILABLE and len(processed_audio) > sr * 0.5:
            logger.debug(f"[{process_id_short}] Applying noise reduction...")
            try:
                processed_audio = noisereduce.reduce_noise(y=processed_audio, sr=sr, stationary=False, prop_decrease=0.85, n_fft=512, hop_length=128, n_std_thresh_stationary=1.5)
                logger.info(f"[{process_id_short}] Noise reduction applied. Length after NR: {len(processed_audio)/sr:.2f}s")
            except Exception as e_nr: logger.warning(f"[{process_id_short}] Noise reduction failed: {e_nr}. Using VAD/original.")
        elif not NOISEREDUCE_AVAILABLE: logger.debug(f"[{process_id_short}] Noisereduce library not available, skipping.")
        
        peak_val = np.abs(processed_audio).max()
        if peak_val > 1e-5: processed_audio = (processed_audio / peak_val) * 0.95
        else: logger.warning(f"[{process_id_short}] Processed reference audio is near silent after VAD/NR. Cloning might be poor.");
        logger.debug(f"[{process_id_short}] Reference audio normalized.")
        return processed_audio

    def _generate_base_tts_audio_with_polly(self, text: str, lang_code_app: str, temp_audio_dir: Path, use_ssml: bool = False) -> Optional[Path]:
        polly_voice_config = self.polly_voice_map.get(lang_code_app)
        if not polly_voice_config: logger.warning(f"No Polly voice config for '{lang_code_app}'."); return None
        polly_client = self._get_polly_client(polly_voice_config['Region'])
        if not polly_client: logger.warning(f"Polly client for '{polly_voice_config['Region']}' unavailable."); return None
        logger.info(f"Attempting Polly TTS for '{lang_code_app}', VoiceId '{polly_voice_config['VoiceId']}', Engine '{polly_voice_config['Engine']}'. SSML: {use_ssml}")
        req_id = str(uuid.uuid4())[:8]
        mp3_path = temp_audio_dir / f"base_polly_{lang_code_app}_{req_id}.mp3"
        wav_path = temp_audio_dir / f"base_polly_{lang_code_app}_{req_id}_16k.wav"
        try:
            req_params = {'Text': text, 'OutputFormat': 'mp3', 'VoiceId': polly_voice_config['VoiceId'], 'Engine': polly_voice_config['Engine']}
            if use_ssml:
                req_params['TextType'] = 'ssml'
                if not text.strip().lower().startswith("<speak>"): req_params['Text'] = f"<speak>{text}</speak>"
            logger.debug(f"Polly request params: {req_params}")
            resp = polly_client.synthesize_speech(**req_params)
            if 'AudioStream' in resp:
                with open(mp3_path, 'wb') as f: f.write(resp['AudioStream'].read())
                if not mp3_path.exists() or mp3_path.stat().st_size < 100: logger.error(f"Polly MP3 small/missing '{lang_code_app}'."); return None
                AudioSegment.from_mp3(str(mp3_path)).set_frame_rate(16000).set_channels(1).export(str(wav_path), format="wav")
                if not wav_path.exists() or wav_path.stat().st_size < 1000: logger.error(f"Polly WAV conversion failed '{lang_code_app}'."); return None
                logger.info(f"Polly 16kHz WAV OK '{lang_code_app}': {wav_path.name} (Size: {wav_path.stat().st_size}b)")
                try: os.remove(mp3_path) 
                except: pass
                return wav_path
            else: logger.error(f"Polly no AudioStream '{lang_code_app}'. Resp: {resp}"); return None
        except boto3.exceptions.Boto3Error as e: # type: ignore
             logger.error(f"Boto3/Polly API Error '{lang_code_app}': {e}", exc_info=False)
             if hasattr(e, 'response') and e.response and 'Error' in e.response: # type: ignore
                 logger.error(f"AWS Error Code: {e.response['Error'].get('Code')}, Message: {e.response['Error'].get('Message')}") # type: ignore
             return None
        except Exception as e: logger.error(f"Generic Polly TTS failure '{lang_code_app}': {e}", exc_info=True); return None

    def _generate_base_tts_audio_with_gtts(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        text_for_gtts = text
        if "<speak>" in text.lower() and "</speak>" in text.lower():
            text_for_gtts = re.sub(r'(?i)<\s*speak\s*>', '', text_for_gtts)
            text_for_gtts = re.sub(r'(?i)<\s*/\s*speak\s*>', '', text_for_gtts)
            text_for_gtts = re.sub(r'(?i)<\s*break\s+time\s*=\s*"[^"]*"\s*/?\s*>', ' ', text_for_gtts)
            text_for_gtts = re.sub(r'\s+', ' ', text_for_gtts).strip()
            if text_for_gtts != text: logger.warning(f"SSML stripped for gTTS. Orig: '{text[:70]}...', New: '{text_for_gtts[:70]}...'")
        logger.warning(f"Fallback gTTS for lang '{lang_code_app}'. Text: '{text_for_gtts[:70]}...'")
        try: from gtts import gTTS # type: ignore
        except ImportError: logger.error("gTTS not installed."); return None
        gtts_code = self._get_simple_lang_code(lang_code_app); req_id = str(uuid.uuid4())[:8]
        mp3_path = temp_audio_dir / f"base_gtts_{lang_code_app}_{req_id}_fb.mp3"
        wav_path = temp_audio_dir / f"base_gtts_{lang_code_app}_{req_id}_16k_fb.wav"
        try:
            gTTS(text=text_for_gtts, lang=gtts_code, slow=False).save(str(mp3_path))
            if not mp3_path.exists() or mp3_path.stat().st_size == 0: logger.error(f"gTTS MP3 failed '{lang_code_app}'."); return None
            AudioSegment.from_mp3(str(mp3_path)).set_frame_rate(16000).set_channels(1).export(str(wav_path), format="wav")
            if not wav_path.exists() or wav_path.stat().st_size < 1000: logger.error(f"gTTS WAV conversion failed '{lang_code_app}'."); return None
            logger.info(f"gTTS WAV for '{lang_code_app}': {wav_path.name}");
            try: os.remove(mp3_path)
            except: pass
            return wav_path
        except Exception as e: logger.error(f"gTTS fallback failed '{lang_code_app}': {e}", exc_info=True); return None

    def _generate_base_tts_audio(self, text: str, lang_code_app: str, temp_audio_dir: Path, use_ssml: bool = False) -> Optional[Path]:
        logger.debug(f"[_generate_base_tts_audio] lang '{lang_code_app}', SSML: {use_ssml}, text: '{text[:70]}...'")
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map:
            polly_path = self._generate_base_tts_audio_with_polly(text, lang_code_app, temp_audio_dir, use_ssml=use_ssml)
            if polly_path and polly_path.exists(): return polly_path
            logger.warning(f"Polly TTS failed '{lang_code_app}' (SSML={use_ssml}). Fallback to gTTS...")
        else:
            if not BOTO3_AVAILABLE: logger.info(f"PollySDK not available. Trying gTTS for '{lang_code_app}'.")
            else: logger.info(f"Polly voice not in map for '{lang_code_app}'. Trying gTTS.")
        gtts_path = self._generate_base_tts_audio_with_gtts(text, lang_code_app, temp_audio_dir)
        if gtts_path and gtts_path.exists(): return gtts_path
        logger.error(f"All TTS failed for lang '{lang_code_app}'."); return None

    def _clone_voice_with_api(self, ref_voice_audio_path: str, target_content_audio_path: str, output_cloned_audio_path: str, req_id_short: str = "clone") -> bool:
        logger.info(f"[{req_id_short}] _clone_voice_with_api: Ref='{Path(ref_voice_audio_path).name}', Content='{Path(target_content_audio_path).name}'")
        if not check_openvoice_api(): logger.warning(f"[{req_id_short}] OpenVoice API NOT available. Skipping cloning."); return False
        if not Path(ref_voice_audio_path).exists() or Path(ref_voice_audio_path).stat().st_size < 1000: logger.error(f"[{req_id_short}] Ref audio MISSING/small: {ref_voice_audio_path}"); return False
        if not Path(target_content_audio_path).exists() or Path(target_content_audio_path).stat().st_size < 1000: logger.error(f"[{req_id_short}] Target TTS audio MISSING/small: {target_content_audio_path}"); return False
        try:
            with open(ref_voice_audio_path, "rb") as f_ref, open(target_content_audio_path, "rb") as f_tgt:
                files = {"reference_audio_file": (Path(ref_voice_audio_path).name, f_ref, "audio/wav"), "content_audio_file": (Path(target_content_audio_path).name, f_tgt, "audio/wav")}
                logger.debug(f"[{req_id_short}] Sending to OpenVoice API /clone-voice. Keys: {list(files.keys())}")
                response = requests.post(f"{OPENVOICE_API_URL}/clone-voice", files=files, timeout=180)
            logger.info(f"[{req_id_short}] OpenVoice API /clone-voice status: {response.status_code}")
            if response.status_code == 200:
                with open(output_cloned_audio_path, "wb") as f_out: f_out.write(response.content)
                if Path(output_cloned_audio_path).exists() and Path(output_cloned_audio_path).stat().st_size > 1000:
                    logger.info(f"[{req_id_short}] OpenVoice cloning OK. Output: {output_cloned_audio_path}"); return True
                else: logger.error(f"[{req_id_short}] OpenVoice API OK, but output file problematic: {output_cloned_audio_path} (Size: {Path(output_cloned_audio_path).stat().st_size if Path(output_cloned_audio_path).exists() else 'N/A'})"); return False
            else:
                err_detail = f"OpenVoice API Error {response.status_code}"
                try: err_data = response.json(); err_detail += f" - Detail: {err_data.get('detail', response.text[:200])}"
                except: err_detail += f" - Response: {response.text[:200] if response.text else '(empty)'}"
                logger.error(f"[{req_id_short}] {err_detail}"); return False
        except requests.exceptions.RequestException as e: logger.error(f"[{req_id_short}] RequestException OpenVoice API: {e}", exc_info=True); return False
        except Exception as e: logger.error(f"[{req_id_short}] General exception _clone_voice_with_api: {e}", exc_info=True); return False

    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str = "eng", target_lang: str = "fra") -> Dict[str, Any]:
        process_id_short = str(time.time_ns())[-6:]
        logger.info(f"[{process_id_short}] CascadedBackend.translate_speech CALLED. App Source: '{source_lang}', App Target: '{target_lang}'")
        if not self.initialized:
            logger.info(f"[{process_id_short}] Backend not initialized, attempting to initialize now...")
            try: self.initialize()
            except Exception as e_init:
                 logger.error(f"[{process_id_short}] CRITICAL: Backend failed to initialize: {e_init}", exc_info=True)
                 return {"audio": torch.zeros((1,16000),dtype=torch.float32), "transcripts": {"source":"INIT_ERROR","target":"INIT_ERROR"}}
            if not self.initialized:
                 logger.error(f"[{process_id_short}] CRITICAL: Backend STILL not initialized after attempt.")
                 return {"audio": torch.zeros((1,16000),dtype=torch.float32), "transcripts": {"source":"INIT_FAIL","target":"INIT_FAIL"}}

        start_time_translate_speech = time.time()
        with tempfile.TemporaryDirectory(prefix=f"cascaded_s2st_{process_id_short}_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            logger.info(f"[{process_id_short}] Using temp directory: {temp_dir}")
            source_text_raw, source_pauses = "ASR_FAILED_OR_SILENT", []
            target_text_raw = "TRANSLATION_FAILED_OR_NO_INPUT"
            output_tensor = torch.zeros((1, 16000), dtype=torch.float32)
            original_ref_for_eval_path = None 
            cloned_audio_for_eval_path_temp = None # Temp path for cloned audio before it's potentially cleaned up

            try:
                asr_start_time = time.time()
                audio_numpy = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
                source_text_raw, source_pauses = self._get_text_and_pauses_from_asr(audio_numpy, source_lang, process_id_short)
                logger.info(f"[{process_id_short}] ASR time: {(time.time()-asr_start_time):.2f}s. Text: '{source_text_raw[:70]}...'. Pauses: {len(source_pauses)}")

                translation_start_time = time.time()
                if not source_text_raw.strip() or source_text_raw == "ASR_MODEL_UNAVAILABLE":
                    target_text_raw = "" if not source_text_raw.strip() else "TRANSLATION_SKIPPED_NO_ASR"
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
                    logger.warning(f"[{process_id_short}] Target text empty/error. TTS silent.")
                else:
                    base_tts_audio_path = self._generate_base_tts_audio(target_text_for_tts, target_lang, temp_dir, use_ssml=is_ssml_text)
                if base_tts_audio_path and base_tts_audio_path.exists():
                    y_base_tts, _ = librosa.load(str(base_tts_audio_path), sr=16000, mono=True)
                    output_tensor = torch.from_numpy(y_base_tts.astype(np.float32)).unsqueeze(0)
                    logger.info(f"[{process_id_short}] Base TTS: {base_tts_audio_path.name} ({(time.time()-tts_start_time):.2f}s)")
                else: logger.error(f"[{process_id_short}] Base TTS failed. Output silent. ({(time.time()-tts_start_time):.2f}s)")

                current_ov_api_available = check_openvoice_api()
                base_tts_ok = bool(base_tts_audio_path and base_tts_audio_path.exists())
                source_text_ok = bool(source_text_raw.strip() and source_text_raw not in ["ASR_FAILED_OR_SILENT", "ASR_MODEL_UNAVAILABLE"])
                should_attempt_cloning = self.use_voice_cloning_config and current_ov_api_available and base_tts_ok and source_text_ok
                logger.info(f"[{process_id_short}] Cloning decision: Config={self.use_voice_cloning_config}, API={current_ov_api_available}, TTS_OK={base_tts_ok}, SrcTxt_OK={source_text_ok} -> Clone={should_attempt_cloning}")

                if should_attempt_cloning:
                    clone_t_start = time.time()
                    preprocessed_ref_numpy = self._preprocess_reference_audio(audio_numpy, sr=16000, process_id_short=process_id_short)
                    
                    # Save preprocessed ref for cloning (and potentially for evaluation)
                    ref_path_for_cloning = temp_dir / f"ref_for_clone_{process_id_short}.wav"
                    sf.write(str(ref_path_for_cloning), preprocessed_ref_numpy, 16000)
                    original_ref_for_eval_path = str(ref_path_for_cloning) # Path to the audio actually used as reference
                    logger.info(f"[{process_id_short}] Preprocessed ref audio for cloning: {ref_path_for_cloning.name}")
                    
                    cloned_output_temp_path = temp_dir / f"cloned_final_temp_{process_id_short}.wav"
                    
                    clone_ok = self._clone_voice_with_api(str(ref_path_for_cloning), str(base_tts_audio_path), str(cloned_output_temp_path), req_id_short=process_id_short)
                    
                    if clone_ok and cloned_output_temp_path.exists() and cloned_output_temp_path.stat().st_size > 1000:
                        y_cloned, _ = librosa.load(str(cloned_output_temp_path), sr=16000, mono=True)
                        output_tensor = torch.from_numpy(y_cloned.astype(np.float32)).unsqueeze(0)
                        cloned_audio_for_eval_path_temp = str(cloned_output_temp_path)
                        logger.info(f"[{process_id_short}] Using OV CLONED audio. ({(time.time()-clone_t_start):.2f}s)")
                        
                        # <<< --- EVALUATION STEP (API Call) --- >>>
                        if original_ref_for_eval_path and cloned_audio_for_eval_path_temp:
                            try:
                                files_to_compare = {
                                    'original_audio': (Path(original_ref_for_eval_path).name, open(original_ref_for_eval_path, 'rb'), 'audio/wav'),
                                    'cloned_audio': (Path(cloned_audio_for_eval_path_temp).name, open(cloned_audio_for_eval_path_temp, 'rb'), 'audio/wav')
                                }
                                logger.info(f"[{process_id_short}] Calling voice similarity API at {VOICE_SIMILARITY_API_URL}/compare-voices/")
                                api_response = requests.post(f"{VOICE_SIMILARITY_API_URL}/compare-voices/", files=files_to_compare, timeout=60)
                                
                                files_to_compare['original_audio'][1].close()
                                files_to_compare['cloned_audio'][1].close()

                                if api_response.status_code == 200:
                                    similarity_data = api_response.json()
                                    similarity_score = similarity_data.get("similarity_score")
                                    if similarity_score is not None:
                                        logger.info(f"[{process_id_short}] VOICE SIMILARITY SCORE (via API): {similarity_score:.4f}")
                                    else:
                                        logger.warning(f"[{process_id_short}] Similarity API response missing score: {similarity_data}")
                                else:
                                    logger.error(f"[{process_id_short}] Voice similarity API call failed with status {api_response.status_code}: {api_response.text[:200]}")
                            except requests.exceptions.RequestException as e_api_call:
                                logger.error(f"[{process_id_short}] RequestException calling voice similarity API: {e_api_call}", exc_info=True)
                            except Exception as e_sim_eval:
                                logger.error(f"[{process_id_short}] Error during voice similarity evaluation via API: {e_sim_eval}", exc_info=True)
                        # <<< --- END OF EVALUATION STEP --- >>>
                    else: 
                        logger.warning(f"[{process_id_short}] OV cloning FAILED/output invalid. Using base TTS. ({(time.time()-clone_t_start):.2f}s)")
                else: logger.info(f"[{process_id_short}] OV cloning SKIPPED.")
                logger.info(f"[{process_id_short}] Final audio shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
            except Exception as e:
                logger.error(f"[{process_id_short}] Error in translate_speech pipeline: {e}", exc_info=True)
                output_tensor = torch.zeros((1, 16000),dtype=torch.float32)
                source_text_raw = source_text_raw if source_text_raw not in ["ASR_FAILED_OR_SILENT", "ASR_MODEL_UNAVAILABLE"] else "PIPELINE_ASR_ERROR"
                target_text_raw = "PIPELINE_ERROR"
            logger.info(f"[{process_id_short}] translate_speech completed in {time.time() - start_time_translate_speech:.2f}s")
            return {"audio": output_tensor, "transcripts": {"source": source_text_raw, "target": target_text_raw}}

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
                if not (BOTO3_AVAILABLE and code_gtts in self.polly_voice_map):
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