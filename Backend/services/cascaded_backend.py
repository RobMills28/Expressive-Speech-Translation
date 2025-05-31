# services/cascaded_backend.py
import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import tempfile
import soundfile as sf
import librosa
import time
import traceback
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
import requests
import json
import sys
import uuid
import re
from scipy.signal import butter, lfilter # For band-pass filter

# --- Optional Imports for Advanced Preprocessing ---
try:
    import noisereduce
    NOISEREDUCE_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported 'noisereduce' library.")
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logging.getLogger(__name__).warning("'noisereduce' library not found. Basic noise reduction will be used if enabled.")

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported 'webrtcvad' library for VAD.")
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logging.getLogger(__name__).warning("'webrtcvad' library not found. Using basic silence removal for VAD.")

DEEPFILTERNET_AVAILABLE = False
df_enhance_func = None
df_init_func = None
try:
    from df.enhance import enhance as df_enhance_imported
    from df.utils import init_df as df_init_imported
    df_enhance_func = df_enhance_imported
    df_init_func = df_init_imported
    DEEPFILTERNET_AVAILABLE = True
    logging.getLogger(__name__).info("Successfully imported 'df' (DeepFilterNet) library components.")
except ImportError:
    logging.getLogger(__name__).warning("'df' (DeepFilterNet) library not found. DeepFilterNet enhancement will be skipped.")
except Exception as e_df_import:
    logging.getLogger(__name__).warning(f"Could not import DeepFilterNet components due to: {e_df_import}. DeepFilterNet enhancement will be skipped.")


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
    test_client.describe_voices(LanguageCode='en-US')
    polly_init_logger.info(f"Boto3 Polly client test successful in region '{aws_default_region}'.")
    BOTO3_AVAILABLE = True
except ImportError:
    polly_init_logger.error("IMPORT ERROR: Failed to import 'boto3'. AWS Polly will be unavailable.")
    BOTO3_AVAILABLE = False
    boto3 = None
except Exception as e:
    polly_init_logger.error(f"BOTO3 CLIENT OR CONFIGURATION FAILURE: {type(e).__name__} - {e}", exc_info=False)
    polly_init_logger.error("DETAILS: This usually means AWS credentials or region are not correctly configured, or network/IAM permission issues.")
    BOTO3_AVAILABLE = False
    if 'boto3' not in sys.modules:
        boto3 = None


logger = logging.getLogger(__name__)
try:
    import whisper # type: ignore
    WHISPER_AVAILABLE_FLAG = True; logger.info("Whisper library FOUND.")
except ImportError: logger.warning("Whisper library NOT FOUND. Falling back."); WHISPER_AVAILABLE_FLAG = False; whisper = None
OPENVOICE_API_URL = os.getenv("OPENVOICE_API_URL", "http://localhost:8000")
VOICE_SIMILARITY_API_URL = os.getenv("VOICE_SIMILARITY_API_URL", "http://localhost:8001")
SAVE_DEBUG_AUDIO_FILES = os.getenv("SAVE_DEBUG_AUDIO_FILES", "false").lower() == "true"
TARGET_LUFS = -23.0 # Target loudness for normalization

OPENVOICE_API_AVAILABLE = False
def check_openvoice_api():
    global OPENVOICE_API_AVAILABLE
    try:
        status_url = f"{OPENVOICE_API_URL}/status"; logger.debug(f"Checking OpenVoice API: {status_url}")
        response = requests.get(status_url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("tone_converter_model_loaded") and data.get("default_source_se_loaded"):
                logger.info(f"OpenVoice API available, models loaded: {data.get('message', '')}"); OPENVOICE_API_AVAILABLE = True; return True
            else: logger.warning(f"OpenVoice API up ({response.status_code}), but models NOT loaded: {data}"); OPENVOICE_API_AVAILABLE = False; return False
        elif response.status_code == 503: logger.warning(f"OpenVoice API /status 503: Service unavailable. Detail: {response.text[:200]}"); OPENVOICE_API_AVAILABLE = False; return False
        else: logger.warning(f"OpenVoice API status check failed HTTP {response.status_code}. Detail: {response.text[:200]}"); OPENVOICE_API_AVAILABLE = False; return False
    except json.JSONDecodeError: logger.warning(f"OpenVoice API /status 200 but not valid JSON: {response.text[:200]}");OPENVOICE_API_AVAILABLE = False; return False
    except requests.exceptions.RequestException as e: logger.warning(f"OpenVoice API unavailable (RequestException): {type(e).__name__} - {e}"); OPENVOICE_API_AVAILABLE = False; return False
    except Exception as e_gen: logger.warning(f"OpenVoice API status check unexpected error: {type(e_gen).__name__} - {e_gen}");OPENVOICE_API_AVAILABLE = False; return False
    OPENVOICE_API_AVAILABLE = False; return False
check_openvoice_api() # Initial check
from .translation_strategy import TranslationBackend
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline

class CascadedBackend(TranslationBackend):
    def __init__(self, device=None, use_voice_cloning=True):
        logger.info(f"Initializing CascadedBackend. Device: {device}, VC Config: {use_voice_cloning}, Polly: {BOTO3_AVAILABLE}, Whisper: {WHISPER_AVAILABLE_FLAG}, NR: {NOISEREDUCE_AVAILABLE}, VAD: {WEBRTCVAD_AVAILABLE}, DFN: {DEEPFILTERNET_AVAILABLE}")
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"CascadedBackend ML device: {self.device}")
        self.initialized = False
        self.use_voice_cloning_config = use_voice_cloning
        self.polly_voice_map = {'eng':{'VoiceId':'Joanna','Engine':'neural','Region':'us-east-1'},'deu':{'VoiceId':'Vicki','Engine':'neural','Region':'eu-central-1'},'spa':{'VoiceId':'Lucia','Engine':'neural','Region':'eu-west-1'},'fra':{'VoiceId':'Lea','Engine':'neural','Region':'eu-west-1'},'ita':{'VoiceId':'Bianca','Engine':'neural','Region':'eu-central-1'},'jpn':{'VoiceId':'Kazuha','Engine':'neural','Region':'ap-northeast-1'},'kor':{'VoiceId':'Seoyeon','Engine':'neural','Region':'ap-northeast-2'},'por':{'VoiceId':'Ines','Engine':'neural','Region':'eu-west-1'},'rus':{'VoiceId':'Olga','Engine':'neural','Region':'eu-central-1'},'cmn':{'VoiceId':'Zhiyu','Engine':'neural','Region':'ap-northeast-1'},'ara':{'VoiceId':'Hala','Engine':'neural','Region':'me-south-1'},'hin':{'VoiceId':'Kajal','Engine':'neural','Region':'ap-south-1'},'nld':{'VoiceId':'Laura','Engine':'neural','Region':'eu-west-1'},'pol':{'VoiceId':'Ewa','Engine':'neural','Region':'eu-central-1'},'tur':{'VoiceId':'Burcu','Engine':'neural','Region':'eu-central-1'},'ukr':{'VoiceId':'Olga','Engine':'neural','Region':'eu-central-1'}}
        self.simple_lang_code_map = { 'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it', 'por': 'pt', 'cmn': 'zh-cn', 'jpn': 'ja', 'kor': 'ko', 'ara': 'ar', 'hin': 'hi', 'nld': 'nl', 'rus': 'ru', 'pol': 'pl', 'tur': 'tr', 'ukr': 'uk'}
        self.display_language_names = {'eng':'English (Joanna, Neural)','deu':'German (Vicki, Neural)','spa':'Spanish (Lucia, Neural)','fra':'French (Lea, Neural)','ita':'Italian (Bianca, Neural)','jpn':'Japanese (Kazuha, Neural)','kor':'Korean (Seoyeon, Neural)','por':'Portuguese (Ines, Eur., Neural)','rus':'Russian (Olga, Neural)','cmn':'Mandarin (Zhiyu, Neural)','ara':'Arabic (Hala, Egy., Neural)','hin':'Hindi (Kajal, Neural)','nld':'Dutch (Laura, Neural)','pol':'Polish (Ewa, Neural)','tur':'Turkish (Burcu, Neural)','ukr':'Ukrainian (Fallback Olga, Rus., Neural)'}
        self.asr_model = None; self.asr_pipeline = None; self.translator_model = None; self.translator_tokenizer = None; self.polly_clients_cache = {}
        self.vad_mode = 3 # Aggressiveness for WebRTCVAD (0-3, 3 is most aggressive)
        if WEBRTCVAD_AVAILABLE: self.vad = webrtcvad.Vad(); self.vad.set_mode(self.vad_mode)
        else: self.vad = None
        self.df_model = None; self.df_state = None
        logger.info(f"CascadedBackend __init__ flags: Whisper={WHISPER_AVAILABLE_FLAG}, OpenVoiceAPI={OPENVOICE_API_AVAILABLE}, PollySDK={BOTO3_AVAILABLE}")
        if SAVE_DEBUG_AUDIO_FILES: logger.info("SAVE_DEBUG_AUDIO_FILES is enabled. Intermediate audio files will be saved.")

    def _get_polly_client(self, region_name: str) -> Optional[boto3.client]: # type: ignore
        if not BOTO3_AVAILABLE or not boto3: logger.warning("Boto3 SDK not available."); return None
        if region_name in self.polly_clients_cache: return self.polly_clients_cache[region_name]
        try:
            logger.info(f"Creating Polly client for region: {region_name}"); client = boto3.client('polly', region_name=region_name)
            self.polly_clients_cache[region_name] = client; return client
        except Exception as e: logger.error(f"Failed to create Polly client for {region_name}: {e}", exc_info=True); return None

    def initialize(self):
        if self.initialized: return
        logger.info(f"Initializing CascadedBackend components on device: {self.device}")
        start_time = time.time()
        try:
            if WHISPER_AVAILABLE_FLAG and whisper:
                self.asr_model = whisper.load_model("medium", device=self.device); logger.info("Whisper ASR loaded ('medium').")
            else:
                self.asr_pipeline = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if self.device.type=='cuda' else -1); logger.info("Transformers ASR pipeline loaded.")
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(self.device)
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            logger.info("NLLB translation model and tokenizer loaded.")
            if BOTO3_AVAILABLE: self._get_polly_client(os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

            if DEEPFILTERNET_AVAILABLE and df_init_func is not None:
                try:
                    logger.info("Attempting to initialize DeepFilterNet model (default)...")
                    self.df_model, self.df_state, _ = df_init_func(log_level="INFO")
                    logger.info("DeepFilterNet model initialized successfully.")
                except Exception as e_df:
                    logger.error(f"Failed to initialize DeepFilterNet: {e_df}", exc_info=True)
                    self.df_model = None; self.df_state = None

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
        text_segments = []; pauses_info: List[Dict[str, Any]] = []; last_word_end_time = 0.0
        if np.abs(audio_numpy).max() < 1e-5: logger.info(f"[{process_id_short}] Input audio near silent."); return "", []
        if WHISPER_AVAILABLE_FLAG and self.asr_model:
            lang_hint = self._get_simple_lang_code(source_lang)
            logger.info(f"[{process_id_short}] Whisper ASR with word_timestamps=True, lang_hint='{lang_hint}'")
            asr_result = self.asr_model.transcribe(audio_numpy, language=lang_hint if lang_hint != 'auto' else None, task="transcribe", fp16=(self.device.type == 'cuda'), word_timestamps=True)
            if "segments" in asr_result:
                current_global_word_index = -1
                for segment_idx, segment in enumerate(asr_result["segments"]):
                    segment_word_list = []
                    if "words" in segment and segment["words"]:
                        for word_info_idx, item in enumerate(segment["words"]):
                            current_global_word_index += 1
                            word_text, start_time, end_time = (item.get("word","").strip(), item.get("start",0.0), item.get("end",0.0)) if isinstance(item,dict) else (item.word.strip(), item.start, item.end)
                            if current_global_word_index > 0 and start_time > last_word_end_time and (pause_duration := start_time - last_word_end_time) > 0.250:
                                pauses_info.append({"start":round(last_word_end_time,3),"end":round(start_time,3),"duration":round(pause_duration,3),"insert_after_word_index":current_global_word_index-1})
                                logger.debug(f"[{process_id_short}] Pause: {pause_duration:.3f}s after word_idx {current_global_word_index-1} ('{text_segments[-1] if text_segments else '<S>'}')")
                            segment_word_list.append(word_text); last_word_end_time = end_time
                        if segment_word_list: text_segments.extend(segment_word_list)
                    elif "text" in segment and (st := segment.get("text","").strip()):
                        current_global_word_index+=len(st.split()); text_segments.append(st)
                        logger.warning(f"[{process_id_short}] Seg {segment_idx} no word timestamps. Pauses impacted."); last_word_end_time = segment.get("end",last_word_end_time)
            source_text = " ".join(text_segments).strip()
            if not source_text and (full_text := asr_result.get("text","").strip()): source_text=full_text; pauses_info=[]; logger.warning(f"[{process_id_short}] Segments empty, using full ASR text. Pauses lost.")
            logger.info(f"[{process_id_short}] Whisper ASR (Lang: {asr_result.get('language','unk')}): '{source_text[:70]}...'")
            logger.info(f"[{process_id_short}] Detected {len(pauses_info)} sig. pauses.")
            if pauses_info: logger.debug(f"[{process_id_short}] Pauses: {json.dumps(pauses_info, indent=2)}")
        elif self.asr_pipeline:
            with tempfile.NamedTemporaryFile(suffix=".wav",delete=False,dir=Path(tempfile.gettempdir())) as f: sf.write(f.name, audio_numpy, 16000); source_text=self.asr_pipeline(f.name)["text"]
            try: os.unlink(f.name)
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

    def _save_debug_audio(self, audio_data: Union[np.ndarray, AudioSegment], filename: str, debug_audio_dir: Path, sr: Optional[int] = 16000):
        if not SAVE_DEBUG_AUDIO_FILES: return
        try:
            debug_audio_dir.mkdir(parents=True, exist_ok=True)
            file_path = debug_audio_dir / filename
            if isinstance(audio_data, np.ndarray):
                sf.write(str(file_path), audio_data, sr)
            elif isinstance(audio_data, AudioSegment):
                audio_data.export(str(file_path), format="wav")
            logger.debug(f"Saved debug audio: {file_path} (Size: {file_path.stat().st_size if file_path.exists() else 'N/A'})")
        except Exception as e:
            logger.error(f"Failed to save debug audio {filename}: {e}", exc_info=True)

    def _normalize_lufs(self, audio_segment: AudioSegment, target_lufs: float = TARGET_LUFS) -> AudioSegment:
        if not audio_segment or len(audio_segment) == 0:
            logger.warning("Cannot normalize LUFS for empty audio segment.")
            return audio_segment
        try:
            change_in_dbfs = target_lufs - audio_segment.dBFS
            if abs(change_in_dbfs) > 60: 
                logger.warning(f"LUFS norm: Extreme gain change ({change_in_dbfs:.2f} dB) requested. Capping to avoid amplifying silence excessively. Original dBFS: {audio_segment.dBFS:.2f}")
                change_in_dbfs = np.clip(change_in_dbfs, -30.0, 30.0) 
            normalized_segment = audio_segment.apply_gain(change_in_dbfs)
            logger.debug(f"LUFS norm: Original dBFS: {audio_segment.dBFS:.2f}, Target LUFS: {target_lufs}, Applied Gain: {change_in_dbfs:.2f} dB, New dBFS: {normalized_segment.dBFS:.2f}")
            return normalized_segment
        except Exception as e:
            logger.error(f"Error during LUFS normalization: {e}", exc_info=True)
            return audio_segment

    def _butter_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, sr: int, order: int = 5) -> np.ndarray:
        nyq = 0.5 * sr; low = lowcut / nyq; high = highcut / nyq
        if low <= 0 and high >= 1: return data 
        if low <= 0: b, a = butter(order, high, btype='low', analog=False)
        elif high >= 1: b, a = butter(order, low, btype='high', analog=False)
        else: b, a = butter(order, [low, high], btype='band', analog=False)
        y = lfilter(b, a, data); return y

    def _preprocess_reference_audio(self, audio_numpy: np.ndarray, sr: int, process_id_short: str, debug_audio_dir: Optional[Path] = None) -> np.ndarray:
        logger.info(f"[{process_id_short}] Preprocessing reference audio (orig len: {len(audio_numpy)/sr:.2f}s, SR: {sr}Hz)...")
        if debug_audio_dir: self._save_debug_audio(audio_numpy, f"{process_id_short}_00_ref_raw.wav", debug_audio_dir, sr)

        try:
            if audio_numpy.dtype != np.float32: audio_numpy = audio_numpy.astype(np.float32)
            peak = np.abs(audio_numpy).max()
            if peak > 1.0: audio_numpy = audio_numpy / peak
            
            audio_int16 = (audio_numpy * 32767).astype(np.int16)
            audio_segment = AudioSegment(audio_int16.tobytes(), frame_rate=sr, sample_width=audio_int16.dtype.itemsize, channels=1)
        except Exception as e:
            logger.error(f"[{process_id_short}] Error converting numpy to AudioSegment: {e}. Using raw numpy array.", exc_info=True)
            processed_audio_final_np = audio_numpy 
            if debug_audio_dir: self._save_debug_audio(processed_audio_final_np, f"{process_id_short}_00b_ref_raw_fallback.wav", debug_audio_dir, sr)
            return processed_audio_final_np

        if debug_audio_dir: self._save_debug_audio(audio_segment, f"{process_id_short}_01_ref_pydub_initial.wav", debug_audio_dir)
        logger.info(f"[{process_id_short}] Initial pydub segment: Duration={len(audio_segment)/1000:.2f}s, dBFS={audio_segment.dBFS:.2f}")

        try:
            logger.debug(f"[{process_id_short}] Applying band-pass filter (80Hz-7500Hz)...")
            audio_numpy_filtered = self._butter_bandpass_filter(audio_numpy, 80, 7500, sr, order=3)
            audio_int16_filtered = (audio_numpy_filtered * 32767).astype(np.int16)
            audio_segment = AudioSegment(audio_int16_filtered.tobytes(), frame_rate=sr, sample_width=audio_int16_filtered.dtype.itemsize, channels=1)
            if debug_audio_dir: self._save_debug_audio(audio_segment, f"{process_id_short}_02_ref_bandpassed.wav", debug_audio_dir)
            logger.info(f"[{process_id_short}] After band-pass: Duration={len(audio_segment)/1000:.2f}s, dBFS={audio_segment.dBFS:.2f}")
        except Exception as e_filter:
            logger.warning(f"[{process_id_short}] Band-pass filtering failed: {e_filter}. Proceeding with pre-filter audio.")

        vad_applied_audio_segment = audio_segment
        if WEBRTCVAD_AVAILABLE and self.vad:
            logger.debug(f"[{process_id_short}] Applying WebRTCVAD (mode: {self.vad_mode})...")
            frame_duration_ms = 30 
            samples_per_frame = int(sr * frame_duration_ms / 1000)
            
            frames = make_chunks(audio_segment, frame_duration_ms)
            voiced_frames = []
            for i, frame_chunk in enumerate(frames):
                if len(frame_chunk.raw_data) == samples_per_frame * frame_chunk.sample_width:
                    if self.vad.is_speech(frame_chunk.raw_data, sr):
                        voiced_frames.append(frame_chunk)
                elif i == len(frames) -1 :
                     if len(frame_chunk.raw_data) > 0 and self.vad.is_speech(frame_chunk.raw_data.ljust(samples_per_frame * frame_chunk.sample_width, b'\0'), sr):
                         voiced_frames.append(frame_chunk)

            if voiced_frames:
                combined_speech = sum(voiced_frames, AudioSegment.empty()) 
                vad_applied_audio_segment = combined_speech
                logger.info(f"[{process_id_short}] WebRTCVAD applied. Duration after VAD: {len(vad_applied_audio_segment)/1000:.2f}s")
                if debug_audio_dir: self._save_debug_audio(vad_applied_audio_segment, f"{process_id_short}_03_ref_webrtcvad.wav", debug_audio_dir)
            else:
                logger.warning(f"[{process_id_short}] WebRTCVAD found no speech. Using pre-VAD audio.")
        else: 
            logger.debug(f"[{process_id_short}] Using pydub basic silence removal (min_silence_len=500, silence_thresh=-40dBFS relative).")
            silence_thresh_val = audio_segment.dBFS - 16 
            chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=silence_thresh_val, keep_silence=100)
            if chunks:
                vad_applied_audio_segment = sum(chunks, AudioSegment.empty())
                logger.info(f"[{process_id_short}] Pydub silence removal. Duration after: {len(vad_applied_audio_segment)/1000:.2f}s, Threshold used: {silence_thresh_val:.2f} dBFS")
                if debug_audio_dir: self._save_debug_audio(vad_applied_audio_segment, f"{process_id_short}_03_ref_pydub_vad.wav", debug_audio_dir)
            else:
                logger.warning(f"[{process_id_short}] Pydub silence removal found no speech. Using pre-VAD audio.")

        if len(vad_applied_audio_segment) < int(sr * 0.5): 
            logger.warning(f"[{process_id_short}] Audio too short after VAD ({len(vad_applied_audio_segment)/1000:.2f}s). Reverting to pre-VAD audio for enhancement.")
            audio_for_enhancement_segment = audio_segment 
        else:
            audio_for_enhancement_segment = vad_applied_audio_segment

        enhanced_audio_segment = audio_for_enhancement_segment
        enh_input_numpy = np.array(audio_for_enhancement_segment.get_array_of_samples(), dtype=np.float32) / 32768.0

        if DEEPFILTERNET_AVAILABLE and self.df_model and self.df_state and df_enhance_func and len(enh_input_numpy) > sr * 0.1:
            logger.info(f"[{process_id_short}] Applying DeepFilterNet enhancement...")
            try:
                df_input_audio_np, df_input_sr = enh_input_numpy, sr
                if sr != 48000: 
                    df_input_audio_np = librosa.resample(df_input_audio_np, orig_sr=sr, target_sr=48000)
                    df_input_sr = 48000
                df_input_tensor = torch.from_numpy(df_input_audio_np.copy()).float()
                with torch.no_grad():
                    enhanced_df_audio_tensor = df_enhance_func(self.df_model, self.df_state, df_input_tensor, sr=df_input_sr)
                enhanced_df_audio_numpy = enhanced_df_audio_tensor.cpu().numpy()
                final_enhanced_numpy = librosa.resample(enhanced_df_audio_numpy, orig_sr=df_input_sr, target_sr=sr) if df_input_sr != sr else enhanced_df_audio_numpy
                enhanced_int16 = (final_enhanced_numpy * 32767).astype(np.int16)
                enhanced_audio_segment = AudioSegment(enhanced_int16.tobytes(), frame_rate=sr, sample_width=enhanced_int16.dtype.itemsize, channels=1)
                logger.info(f"[{process_id_short}] DeepFilterNet enhancement applied. Duration: {len(enhanced_audio_segment)/1000:.2f}s")
                if debug_audio_dir: self._save_debug_audio(enhanced_audio_segment, f"{process_id_short}_04_ref_deepfilter.wav", debug_audio_dir)
            except Exception as e_df_enh:
                logger.warning(f"[{process_id_short}] DeepFilterNet failed: {e_df_enh}. Proceeding without DFN.", exc_info=True)
        elif NOISEREDUCE_AVAILABLE and len(enh_input_numpy) > sr * 0.5: 
            logger.info(f"[{process_id_short}] Applying noisereduce (DFN not used or failed)...")
            try:
                nr_output_numpy = noisereduce.reduce_noise(y=enh_input_numpy, sr=sr, stationary=False, prop_decrease=0.9, n_fft=1024, hop_length=256, n_std_thresh_stationary=1.5)
                nr_int16 = (nr_output_numpy * 32767).astype(np.int16)
                enhanced_audio_segment = AudioSegment(nr_int16.tobytes(), frame_rate=sr, sample_width=nr_int16.dtype.itemsize, channels=1)
                logger.info(f"[{process_id_short}] Noisereduce applied. Duration: {len(enhanced_audio_segment)/1000:.2f}s")
                if debug_audio_dir: self._save_debug_audio(enhanced_audio_segment, f"{process_id_short}_04_ref_noisereduce.wav", debug_audio_dir)
            except Exception as e_nr:
                logger.warning(f"[{process_id_short}] Noisereduce failed: {e_nr}. Using pre-NR audio.")
        else:
            logger.info(f"[{process_id_short}] No speech enhancement applied (DFN/NR not available or audio too short).")

        normalized_segment = self._normalize_lufs(enhanced_audio_segment, target_lufs=TARGET_LUFS)
        if debug_audio_dir: self._save_debug_audio(normalized_segment, f"{process_id_short}_05_ref_lufs_normalized.wav", debug_audio_dir)
        logger.info(f"[{process_id_short}] Final preprocessed ref: Duration={len(normalized_segment)/1000:.2f}s, dBFS={normalized_segment.dBFS:.2f}")

        # Adjusted minimum length warning thresholds
        min_ideal_len_s = 5.0 
        min_acceptable_len_s = 2.0

        if len(normalized_segment) < int(sr * min_acceptable_len_s): 
            logger.warning(f"[{process_id_short}] Reference audio is VERY short ({len(normalized_segment)/1000:.2f}s, min ideal {min_ideal_len_s}s) after all preprocessing. Cloning quality will likely be poor.")
        elif len(normalized_segment) < int(sr * min_ideal_len_s): 
             logger.info(f"[{process_id_short}] Reference audio is shorter than ideal ({len(normalized_segment)/1000:.2f}s vs ideal {min_ideal_len_s}s). Quality depends on clarity.")
        else:
            logger.info(f"[{process_id_short}] Reference audio length ({len(normalized_segment)/1000:.2f}s) is good.")


        final_numpy_output = np.array(normalized_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        return final_numpy_output.astype(np.float32)

    def _generate_base_tts_audio_with_polly(self, text: str, lang_code_app: str, temp_audio_dir: Path, use_ssml: bool = False) -> Optional[Path]:
        polly_voice_config = self.polly_voice_map.get(lang_code_app);
        if not polly_voice_config: logger.warning(f"No Polly voice config for '{lang_code_app}'."); return None
        polly_client = self._get_polly_client(polly_voice_config['Region'])
        if not polly_client: logger.warning(f"Polly client for '{polly_voice_config['Region']}' unavailable."); return None
        logger.info(f"Attempting Polly TTS for '{lang_code_app}', VoiceId '{polly_voice_config['VoiceId']}', Engine '{polly_voice_config['Engine']}'. SSML: {use_ssml}")
        req_id = str(uuid.uuid4())[:8]; mp3_path = temp_audio_dir / f"base_polly_{lang_code_app}_{req_id}.mp3"; wav_path = temp_audio_dir / f"base_polly_{lang_code_app}_{req_id}_16k.wav"
        try:
            req_params = {'Text': text, 'OutputFormat': 'mp3', 'VoiceId': polly_voice_config['VoiceId'], 'Engine': polly_voice_config['Engine']}
            if use_ssml: req_params['TextType'] = 'ssml';
            if use_ssml and not text.strip().lower().startswith("<speak>"): req_params['Text'] = f"<speak>{text}</speak>"
            logger.debug(f"Polly request params: {req_params}"); resp = polly_client.synthesize_speech(**req_params)
            if 'AudioStream' in resp:
                with open(mp3_path, 'wb') as f: f.write(resp['AudioStream'].read())
                if not mp3_path.exists() or mp3_path.stat().st_size < 100: logger.error(f"Polly MP3 small/missing '{lang_code_app}'."); return None
                AudioSegment.from_mp3(str(mp3_path)).set_frame_rate(16000).set_channels(1).export(str(wav_path), format="wav")
                if not wav_path.exists() or wav_path.stat().st_size < 1000: logger.error(f"Polly WAV conversion failed '{lang_code_app}'."); return None
                logger.info(f"Polly 16kHz WAV OK '{lang_code_app}': {wav_path.name} (Size: {wav_path.stat().st_size}b)");
                try: os.remove(mp3_path)
                except: pass
                return wav_path
            else: logger.error(f"Polly no AudioStream '{lang_code_app}'. Resp: {resp}"); return None
        except boto3.exceptions.Boto3Error as e: 
            logger.error(f"Boto3/Polly API Error for '{lang_code_app}': {type(e).__name__} - {e}", exc_info=False)
            if hasattr(e, 'response') and e.response and 'Error' in e.response:
                logger.error(f"AWS Error Code: {e.response['Error'].get('Code')}, Message: {e.response['Error'].get('Message')}")
            return None
        except Exception as e: logger.error(f"Generic Polly TTS failure '{lang_code_app}': {e}", exc_info=True); return None

    def _generate_base_tts_audio_with_gtts(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        text_for_gtts = text
        if "<speak>" in text.lower() and "</speak>" in text.lower():
            text_for_gtts = re.sub(r'(?i)<\s*speak\s*>', '', text_for_gtts); text_for_gtts = re.sub(r'(?i)<\s*/\s*speak\s*>', '', text_for_gtts)
            text_for_gtts = re.sub(r'(?i)<\s*break\s+time\s*=\s*"[^"]*"\s*/?\s*>', ' ', text_for_gtts); text_for_gtts = re.sub(r'\s+', ' ', text_for_gtts).strip()
            if text_for_gtts != text: logger.warning(f"SSML stripped for gTTS. Orig: '{text[:70]}...', New: '{text_for_gtts[:70]}...'")
        logger.warning(f"Fallback gTTS for lang '{lang_code_app}'. Text: '{text_for_gtts[:70]}...'")
        try: from gtts import gTTS # type: ignore
        except ImportError: logger.error("gTTS not installed."); return None
        gtts_code=self._get_simple_lang_code(lang_code_app); req_id=str(uuid.uuid4())[:8]; mp3_path=temp_audio_dir/f"base_gtts_{lang_code_app}_{req_id}_fb.mp3"; wav_path=temp_audio_dir/f"base_gtts_{lang_code_app}_{req_id}_16k_fb.wav"
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

    def _log_audio_properties(self, audio_path: Path, label: str, process_id_short: str):
        if not audio_path or not audio_path.exists():
            logger.warning(f"[{process_id_short}] Cannot log properties for '{label}', file missing: {audio_path}")
            return
        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=True) 
            duration = librosa.get_duration(y=y, sr=sr)
            peak_amp = np.abs(y).max()
            rms_amp = np.sqrt(np.mean(y**2))
            audio_segment = AudioSegment.from_file(str(audio_path))
            lufs = audio_segment.dBFS 
            logger.info(f"[{process_id_short}] Properties for '{label}' ({audio_path.name}): Duration={duration:.2f}s, SR={sr}Hz, Peak={peak_amp:.4f}, RMS={rms_amp:.4f}, dBFS(LUFS_proxy)={lufs:.2f}")
        except Exception as e:
            logger.error(f"[{process_id_short}] Error logging properties for {label} ({audio_path.name}): {e}", exc_info=True)

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
            debug_audio_storage_dir = temp_dir / "debug_audio"
            if SAVE_DEBUG_AUDIO_FILES: debug_audio_storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{process_id_short}] Using temp directory: {temp_dir}")
            source_text_raw, source_pauses = "ASR_FAILED_OR_SILENT", []
            target_text_raw = "TRANSLATION_FAILED_OR_NO_INPUT"
            output_tensor = torch.zeros((1, 16000), dtype=torch.float32)
            original_ref_for_eval_path = None 
            cloned_audio_for_eval_path_temp = None

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
                    self._log_audio_properties(base_tts_audio_path, "BaseTTS_for_Cloning", process_id_short)
                else: logger.error(f"[{process_id_short}] Base TTS failed. Output silent. ({(time.time()-tts_start_time):.2f}s)")

                current_ov_api_available = check_openvoice_api() 
                base_tts_ok = bool(base_tts_audio_path and base_tts_audio_path.exists())
                source_text_ok = bool(source_text_raw.strip() and source_text_raw not in ["ASR_FAILED_OR_SILENT", "ASR_MODEL_UNAVAILABLE"])
                should_attempt_cloning = self.use_voice_cloning_config and current_ov_api_available and base_tts_ok and source_text_ok
                logger.info(f"[{process_id_short}] Cloning decision: Config={self.use_voice_cloning_config}, API_Available={current_ov_api_available}, BaseTTS_OK={base_tts_ok}, SourceText_OK={source_text_ok} -> AttemptClone={should_attempt_cloning}")

                if should_attempt_cloning:
                    clone_t_start = time.time()
                    preprocessed_ref_numpy = self._preprocess_reference_audio(audio_numpy, sr=16000, process_id_short=process_id_short, debug_audio_dir=debug_audio_storage_dir)
                    
                    ref_path_for_cloning = temp_dir / f"ref_for_clone_{process_id_short}.wav"
                    sf.write(str(ref_path_for_cloning), preprocessed_ref_numpy, 16000)
                    original_ref_for_eval_path = str(ref_path_for_cloning) 
                    logger.info(f"[{process_id_short}] Preprocessed ref audio for cloning: {ref_path_for_cloning.name}")
                    self._log_audio_properties(ref_path_for_cloning, "PreprocessedRef_for_Cloning", process_id_short)
                    
                    cloned_output_temp_path = temp_dir / f"cloned_final_temp_{process_id_short}.wav"
                    
                    clone_ok = self._clone_voice_with_api(str(ref_path_for_cloning), str(base_tts_audio_path), str(cloned_output_temp_path), req_id_short=process_id_short)
                    
                    if clone_ok and cloned_output_temp_path.exists() and cloned_output_temp_path.stat().st_size > 1000:
                        y_cloned, _ = librosa.load(str(cloned_output_temp_path), sr=16000, mono=True)
                        output_tensor = torch.from_numpy(y_cloned.astype(np.float32)).unsqueeze(0)
                        cloned_audio_for_eval_path_temp = str(cloned_output_temp_path) 
                        logger.info(f"[{process_id_short}] Using OV CLONED audio. ({(time.time()-clone_t_start):.2f}s)")
                        self._log_audio_properties(cloned_output_temp_path, "ClonedAudio_from_OpenVoice", process_id_short)
                        
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
        if hasattr(self, 'df_model') and self.df_model: del self.df_model; self.df_model = None; logger.debug("DeepFilterNet model unloaded.")
        if hasattr(self, 'df_state') and self.df_state: del self.df_state; self.df_state = None; logger.debug("DeepFilterNet state unloaded.")
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache(); logger.debug("CUDA cache emptied.")
        import gc; gc.collect()
        logger.info("CascadedBackend resources cleaned.")