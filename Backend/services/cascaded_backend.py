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
from pydub import AudioSegment, silence
import requests
import json
import sys
import uuid
import re
from scipy.signal import butter, lfilter

from .audio_debug_analyzer import AudioDebugAnalyzer
from .temporal_mapper import TemporalMapper
from .translation_strategy import TranslationBackend
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .visual_temporal_mapper import VisualTemporalMapper

logger = logging.getLogger(__name__)

try:
    import noisereduce
    NOISEREDUCE_AVAILABLE = True
    logger.info("Successfully imported 'noisereduce' library.")
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.warning("'noisereduce' library not found. Basic noise reduction will be used if enabled for reference audio.")

SAVE_DEBUG_AUDIO_FILES = os.getenv("SAVE_DEBUG_AUDIO_FILES", "false").lower() == "true"
TARGET_LUFS = -23.0

COSYVOICE_API_URL = os.getenv("COSYVOICE_API_URL", "http://localhost:8002")
VOICE_SIMILARITY_API_URL = os.getenv("VOICE_SIMILARITY_API_URL", "http://localhost:8001")

class CascadedBackend(TranslationBackend):
    def __init__(self, device=None, use_voice_cloning=True):
        logger.info(f"Initializing CascadedBackend (to use CosyVoice API). Device: {device}")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CascadedBackend ML device: {self.device}")
        self.temporal_mapper = TemporalMapper()
        self.visual_temporal_mapper = VisualTemporalMapper()
        self.debug_analyzer = AudioDebugAnalyzer()
        self.initialized = False

        self.simple_lang_code_map_cosy = {
            'eng': 'en', 'spa': 'es', 'fra': 'fr', 'deu': 'de', 'ita': 'it', 'por': 'pt', 'pol': 'pl',
            'tur': 'tr', 'rus': 'ru', 'nld': 'nl', 'ces': 'cs', 'ara': 'ar', 'cmn': 'zh',
            'jpn': 'ja', 'hun': 'hu', 'kor': 'ko', 'hin': 'hi'
        }
        self.display_language_names = {k: name.capitalize() for k, name in self.simple_lang_code_map_cosy.items()}
        self.display_language_names['cmn'] = 'Chinese (Mandarin)'

        if SAVE_DEBUG_AUDIO_FILES: logger.info("SAVE_DEBUG_AUDIO_FILES is enabled.")

    def initialize(self):
        if not self.visual_temporal_mapper.initialize():
           logger.warning("Visual temporal mapper initialization failed, falling back to audio-only mapping")
        
        # --- NEW WARM-UP CODE ---
        logger.info("Performing a warm-up inference on CosyVoice API to ensure models are loaded...")
        try:
            self._warmup_cosyvoice_api()
            logger.info("CosyVoice API warmed up successfully.")
        except Exception as e_warmup:
            logger.error(f"CosyVoice API warm-up failed: {e_warmup}. The service may be slow on the first request.")
            # We don't fail initialization here, just warn the user.
        # --- END OF NEW WARM-UP CODE ---
        try:
            self.initialized = True
            logger.info(f"CascadedBackend connections verified and warmed up successfully in {time.time() - start_time:.2f}s. Models will be loaded on-demand.")
        except Exception as e:
            logger.error(f"Failed to initialize CascadedBackend: {e}", exc_info=True)
            self.initialized = False
            raise

    def _check_cosyvoice_api_status(self) -> bool:
        max_retries = 5
        retry_delay_seconds = 10  # Wait 10 seconds between retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Checking CosyVoice API status at {COSYVOICE_API_URL}/health (Attempt {attempt + 1}/{max_retries})")
                response = requests.get(f"{COSYVOICE_API_URL}/health", timeout=20) # Increased timeout
                
                if response.status_code == 200:
                    health_data = response.json()
                    api_status = health_data.get("status")
                    api_message = health_data.get("message", "")
                    if api_status == "healthy":
                        logger.info(f"SUCCESS: CosyVoice API is healthy: {api_message}")
                        return True
                    else:
                        logger.warning(f"CosyVoice API reported status '{api_status}'. Retrying...")
                else:
                    logger.warning(f"CosyVoice API status check failed with HTTP {response.status_code}. Retrying...")

            except requests.exceptions.RequestException as e_req:
                logger.warning(f"CosyVoice API request error: {e_req}. Retrying in {retry_delay_seconds}s...")

            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)

        logger.error(f"CRITICAL: CosyVoice API did not become healthy after {max_retries} attempts.")
        return False

    def _warmup_cosyvoice_api(self):
        """
        Sends a short, silent audio file and a simple text prompt to the CosyVoice API.
        This forces the API to load all its models into memory before the backend
        reports itself as fully initialized, preventing timeouts on the first real request.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            # Create a 1-second silent WAV file for the reference audio
            silent_audio = np.zeros(16000, dtype=np.float32)
            silent_wav_path = temp_dir_path / "warmup.wav"
            sf.write(str(silent_wav_path), silent_audio, 16000)

            with open(silent_wav_path, 'rb') as ref_audio_file:
                files = {'reference_speaker_wav': ref_audio_file}
                data = {
                    'text_to_synthesize': 'Hello world.',
                    'target_language_code': 'en'
                }
                # Use a long timeout for this first call
                response = requests.post(f"{COSYVOICE_API_URL}/generate-speech/", files=files, data=data, timeout=300)
                response.raise_for_status() # Raise an exception if the warmup fails

    def _convert_to_nllb_code(self, lang_code_app: str) -> str:
        mapping = {'eng':'eng_Latn','fra':'fra_Latn','spa':'spa_Latn','deu':'deu_Latn','ita':'ita_Latn','por':'por_Latn','rus':'rus_Cyrl','cmn':'zho_Hans','jpn':'jpn_Jpan','kor':'kor_Hang','ara':'ara_Arab','hin':'hin_Deva','nld':'nld_Latn','pol':'pol_Latn','tur':'tur_Latn','ukr':'ukr_Cyrl','ces':'ces_Latn','hun':'hun_Latn'}
        return mapping.get(lang_code_app.lower(), 'eng_Latn')

    def _get_cosyvoice_lang_code(self, lang_code_app: str) -> str:
        return self.simple_lang_code_map_cosy.get(lang_code_app.lower(), 'en')

    def _get_text_and_pauses_from_asr(self, audio_numpy: np.ndarray, source_lang: str, process_id_short: str, asr_model) -> tuple[str, List[Dict[str, Any]], Optional[List[Dict[str,float]]]]:
        text_segments_list = []; pauses_info: List[Dict[str, Any]] = []; last_word_end_time = 0.0
        word_level_timestamps: Optional[List[Dict[str,float]]] = []

        if np.abs(audio_numpy).max() < 1e-5:
            return "", [], []
        if not asr_model:
            return "ASR_MODEL_UNAVAILABLE", [], []
        
        simple_source_lang = source_lang[:2] if source_lang else None
        logger.info(f"[{process_id_short}] Whisper ASR with word_timestamps=True, lang_hint='{simple_source_lang or 'auto'}'")
        import whisper
        asr_result = asr_model.transcribe(audio_numpy, language=simple_source_lang, task="transcribe", fp16=(self.device.type == 'cuda' if isinstance(self.device, torch.device) else self.device == 'cuda'), word_timestamps=True)

        full_text_parts = []
        if "segments" in asr_result:
            current_global_word_index = -1
            for segment in asr_result["segments"]:
                segment_text_parts = []
                if "words" in segment and segment["words"]:
                    for word_info in segment["words"]:
                        current_global_word_index += 1
                        word, start, end = word_info.get("word", "").strip(), word_info.get("start", 0.0), word_info.get("end", 0.0)
                        word_level_timestamps.append({"word": word, "start": start, "end": end})
                        if current_global_word_index > 0 and start > last_word_end_time:
                            if (pause_duration := start - last_word_end_time) > 0.250:
                                pauses_info.append({"start": round(last_word_end_time, 3), "end": round(start, 3), "duration": round(pause_duration, 3), "insert_after_word_index": current_global_word_index -1})
                        segment_text_parts.append(word)
                        last_word_end_time = end
                elif "text" in segment:
                    segment_text_parts.append(segment.get("text", "").strip())
                    last_word_end_time = segment.get("end", last_word_end_time)

                if segment_text_parts:
                    full_text_parts.append(" ".join(segment_text_parts))

        source_text = " ".join(full_text_parts).strip()
        if not source_text and asr_result.get("text", "").strip():
            source_text = asr_result.get("text", "").strip()
            pauses_info, word_level_timestamps = [], []
            logger.warning(f"[{process_id_short}] No text from segments, using full ASR text. Pause info lost.")

        detected_lang = asr_result.get('language', 'unknown')
        logger.info(f"[{process_id_short}] Whisper ASR (Detected Lang: {detected_lang}): '{source_text[:70]}...'")
        logger.info(f"[{process_id_short}] Detected {len(pauses_info)} significant pauses.")
        return source_text, pauses_info, word_level_timestamps

    def _save_debug_audio(self, audio_data: Union[np.ndarray, AudioSegment], filename: str, debug_audio_dir: Path, sr: Optional[int] = 16000):
        if not SAVE_DEBUG_AUDIO_FILES: return
        try:
            debug_audio_dir.mkdir(parents=True, exist_ok=True)
            file_path = debug_audio_dir / filename
            if isinstance(audio_data, np.ndarray):
                sf.write(str(file_path), audio_data, sr)
            elif isinstance(audio_data, AudioSegment):
                audio_data.export(str(file_path), format="wav")
            logger.debug(f"Saved debug audio: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save debug audio {filename}: {e}", exc_info=True)

    def _log_audio_properties(self, audio_path: Path, label: str, process_id_short: str):
        if not audio_path or not audio_path.exists():
            return
        try:
            y, sr_loaded = librosa.load(str(audio_path), sr=None, mono=True)
            duration, peak_amp = librosa.get_duration(y=y, sr=sr_loaded), np.abs(y).max()
            rms_amp = np.sqrt(np.mean(y**2))
            audio_segment = AudioSegment.from_file(str(audio_path))
            lufs = audio_segment.dBFS
            logger.info(f"[{process_id_short}] Properties for '{label}' ({audio_path.name}): Duration={duration:.2f}s, SR={sr_loaded}Hz, Peak={peak_amp:.4f}, RMS={rms_amp:.4f}, dBFS(LUFS_proxy)={lufs:.2f}")
        except Exception as e:
            logger.error(f"[{process_id_short}] Error logging properties for {label} ({audio_path.name}): {e}", exc_info=True)

    # Complete replacement for the problematic function in cascaded_backend.py
    def _apply_natural_temporal_mapping(self, generated_audio_path: Path, 
                               word_timestamps: List[Dict[str, Any]],
                               source_audio_numpy: np.ndarray,
                               original_video_path: Path,
                               temp_dir: Path, process_id_short: str) -> Path:
        """
        Enhanced temporal mapping with comprehensive debugging
        """
        logger.info(f"[{process_id_short}] Starting ENHANCED temporal mapping with comprehensive debugging")
        
        try:
            # Load the generated audio
            generated_audio, sr = librosa.load(str(generated_audio_path), sr=16000, mono=True)
            original_duration = len(generated_audio) / 16000
            
            logger.info(f"[{process_id_short}] Loaded generated audio: {original_duration:.2f}s, "
                    f"{len(generated_audio)} samples at {sr}Hz")
            
            # Debug: Analyze original generated audio
            debug_dir = temp_dir / "debug_analysis"
            debug_dir.mkdir(exist_ok=True)
            
            logger.info(f"[{process_id_short}] Analyzing original generated audio...")
            original_analysis = self.debug_analyzer.analyze_audio_placement(
                generated_audio, f"{process_id_short}_original", debug_dir
            )
            
            # Try visual-guided mapping if video is provided
            if original_video_path and original_video_path.exists():
                try:
                    logger.info(f"[{process_id_short}] Attempting visual-guided mapping with video: {original_video_path.name}")
                    
                    adjusted_audio = self.visual_temporal_mapper.apply_visual_temporal_mapping(
                        generated_audio, original_video_path, process_id_short
                    )
                    
                    # Debug: Analyze adjusted audio
                    logger.info(f"[{process_id_short}] Analyzing temporally mapped audio...")
                    mapped_analysis = self.debug_analyzer.analyze_audio_placement(
                        adjusted_audio, f"{process_id_short}_mapped", debug_dir
                    )
                    
                    # Compare before and after
                    comparison = self.debug_analyzer.compare_before_after(
                        generated_audio, adjusted_audio, process_id_short, debug_dir
                    )
                    
                    adjusted_duration = len(adjusted_audio) / 16000
                    duration_change = adjusted_duration - original_duration
                    
                    logger.info(f"[{process_id_short}] Visual mapping SUCCESS: "
                            f"original={original_duration:.2f}s → adjusted={adjusted_duration:.2f}s "
                            f"(change: {duration_change:+.2f}s)")
                    
                    # Detailed content analysis
                    if mapped_analysis['has_content']:
                        logger.info(f"[{process_id_short}] Mapped audio content span: "
                                f"{mapped_analysis['content_start_time']:.2f}s - "
                                f"{mapped_analysis['content_end_time']:.2f}s "
                                f"({mapped_analysis['content_duration']:.2f}s content)")
                    else:
                        logger.error(f"[{process_id_short}] CRITICAL: Mapped audio has NO CONTENT!")
                    
                    mapping_method = "visual-guided"
                    
                except Exception as e_visual:
                    logger.error(f"[{process_id_short}] Visual mapping FAILED: {e_visual}", exc_info=True)
                    
                    # Fallback: ensure natural audio flow
                    adjusted_audio = self._ensure_natural_flow(generated_audio, source_audio_numpy)
                    mapping_method = "natural flow fallback"
                    
                    # Debug fallback too
                    fallback_analysis = self.debug_analyzer.analyze_audio_placement(
                        adjusted_audio, f"{process_id_short}_fallback", debug_dir
                    )
                    
                    adjusted_duration = len(adjusted_audio) / 16000
                    logger.info(f"[{process_id_short}] Fallback mapping: {adjusted_duration:.2f}s")
            else:
                # No video provided, ensure natural audio flow
                logger.info(f"[{process_id_short}] No video provided, using natural flow only")
                adjusted_audio = self._ensure_natural_flow(generated_audio, source_audio_numpy)
                mapping_method = "natural flow only"
                
                # Debug natural flow
                natural_analysis = self.debug_analyzer.analyze_audio_placement(
                    adjusted_audio, f"{process_id_short}_natural", debug_dir
                )
                
                adjusted_duration = len(adjusted_audio) / 16000
                logger.info(f"[{process_id_short}] Natural flow mapping: {adjusted_duration:.2f}s")
            
            # Save the adjusted audio
            adjusted_audio_path = temp_dir / f"debug_temporally_mapped_{process_id_short}.wav"
            sf.write(str(adjusted_audio_path), adjusted_audio, 16000)
            
            final_duration = len(adjusted_audio) / 16000
            duration_change = final_duration - original_duration
            
            # Final verification
            final_peak = np.max(np.abs(adjusted_audio))
            final_rms = np.sqrt(np.mean(adjusted_audio ** 2))
            
            logger.info(f"[{process_id_short}] TEMPORAL MAPPING COMPLETE ({mapping_method}):")
            logger.info(f"  Duration: {original_duration:.2f}s → {final_duration:.2f}s (change: {duration_change:+.2f}s)")
            logger.info(f"  Final audio peak: {final_peak:.4f}, RMS: {final_rms:.4f}")
            logger.info(f"  Debug files saved in: {debug_dir}")
            logger.info(f"  Final mapped audio: {adjusted_audio_path.name}")
            
            if final_peak < 0.001:
                logger.error(f"[{process_id_short}] ⚠️  CRITICAL WARNING: Final audio appears to be silent!")
            else:
                logger.info(f"[{process_id_short}] ✅ Final audio has content")
            
            return adjusted_audio_path
            
        except Exception as e:
            logger.error(f"[{process_id_short}] CRITICAL ERROR in temporal mapping: {e}", exc_info=True)
            return generated_audio_path

        
    def _ensure_natural_flow(self, audio: np.ndarray, source_audio: np.ndarray) -> np.ndarray:
        """
        Ensure the audio has natural flow based on source characteristics
        """
        # Analyze source audio for basic characteristics
        source_duration = len(source_audio) / self.sample_rate if hasattr(self, 'sample_rate') else len(source_audio) / 16000
        audio_duration = len(audio) / 16000
        
        # If the translated audio is much shorter than source, it might need some natural pauses
        if source_duration > audio_duration * 1.8:
            # Add some natural leading pause (not forced duration matching)
            natural_pause_duration = min(1.0, (source_duration - audio_duration) * 0.2)
            pause_samples = int(natural_pause_duration * 16000)
            
            # Create subtle room tone instead of silence
            if len(source_audio) > 1000:
                # Use beginning of source as room tone template
                room_tone_template = source_audio[:500]
                room_tone_level = np.std(room_tone_template) * 0.05  # Very quiet
                natural_pause = np.random.normal(0, room_tone_level, pause_samples).astype(np.float32)
            else:
                natural_pause = np.zeros(pause_samples, dtype=np.float32)
            
            return np.concatenate([natural_pause, audio])
        
        # Audio duration is reasonable, return as-is
        return audio


    def _get_reference_audio_for_cloning(self, input_audio_numpy_16k: np.ndarray,
                                         process_id_short: str, temp_dir: Path,
                                         debug_audio_storage_dir: Optional[Path]) -> Path:
        logger.info(f"[{process_id_short}] Preparing reference audio for CosyVoice API from input (16kHz).")
        target_sr_for_ref = 16000
        max_ref_len_s = 25
        max_ref_samples = int(max_ref_len_s * target_sr_for_ref)
        current_audio_np = input_audio_numpy_16k

        if current_audio_np.shape[0] > max_ref_samples:
            ref_audio_np = current_audio_np[:max_ref_samples]
            logger.info(f"[{process_id_short}] Using first {max_ref_len_s}s of input as reference for CosyVoice.")
        else:
            ref_audio_np = current_audio_np
            logger.info(f"[{process_id_short}] Using full input ({len(current_audio_np)/target_sr_for_ref:.2f}s) as reference for CosyVoice.")

        ref_audio_path = temp_dir / f"ref_for_cosyvoice_api_{process_id_short}_16k.wav"
        sf.write(str(ref_audio_path), ref_audio_np, target_sr_for_ref, subtype='PCM_16')

        if debug_audio_storage_dir:
             self._save_debug_audio(ref_audio_np, f"{process_id_short}_ref_for_cosyAPI_sent_16k.wav", debug_audio_storage_dir, target_sr_for_ref)
        self._log_audio_properties(ref_audio_path, f"RefAudioForCosyVoiceAPI_16kHz", process_id_short)
        return ref_audio_path

    # Modified translate_speech function - COMPLETE VERSION
    # THIS IS THE NEW, COMPLETE METHOD
    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str = "eng", target_lang: str = "fra", original_video_path: Optional[Path] = None) -> Dict[str, Any]:
        process_id_short = str(time.time_ns())[-6:]
        logger.info(f"[{process_id_short}] CascadedBackend translate_speech started (on-demand loading).")
        
        if not self.initialized:
            raise RuntimeError("Backend not initialized.")

        asr_model = None
        translator_model = None
        translator_tokenizer = None
        source_text_raw = "ASR_FAILED"
        target_text_raw = "TRANSLATION_FAILED"
        word_timestamps = []
        
        # Ensure audio is in the correct format for processing
        audio_numpy_16k = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
        
        try:
            # --- Step 1: Load, Use, and Release ASR Model ---
            logger.info(f"[{process_id_short}] Loading ASR model into memory...")
            import whisper
            asr_model = whisper.load_model("medium", device=self.device)
            
            source_text_raw, _, word_timestamps = self._get_text_and_pauses_from_asr(
                audio_numpy_16k, source_lang, process_id_short, asr_model
            )
            if not source_text_raw.strip():
                raise RuntimeError("ASR failed or produced empty text.")
        finally:
            if asr_model:
                del asr_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info(f"[{process_id_short}] ASR model released from memory.")

        try:
            # --- Step 2: Load, Use, and Release Translation Model ---
            logger.info(f"[{process_id_short}] Loading Translation model into memory...")
            translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(self.device)
            translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

            src_nllb_code = self._convert_to_nllb_code(source_lang)
            tgt_nllb_code = self._convert_to_nllb_code(target_lang)
            
            translator_tokenizer.src_lang = src_nllb_code
            input_ids = translator_tokenizer(source_text_raw, return_tensors="pt").input_ids.to(self.device)
            generated_tokens = translator_model.generate(input_ids, forced_bos_token_id=translator_tokenizer.lang_code_to_id[tgt_nllb_code])
            target_text_raw = translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            if not target_text_raw.strip():
                raise RuntimeError("Translation result was empty.")
        finally:
            if translator_model:
                del translator_model
                del translator_tokenizer
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info(f"[{process_id_short}] Translation model released from memory.")

        # --- Step 3: Call CosyVoice API (already memory-efficient) ---
        with tempfile.TemporaryDirectory(prefix=f"cosy_api_{process_id_short}_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            reference_speaker_wav_path = self._get_reference_audio_for_cloning(audio_numpy_16k, process_id_short, temp_dir, None)
            
            with open(reference_speaker_wav_path, 'rb') as ref_audio_file:
                files = {'reference_speaker_wav': ref_audio_file}
                data = {'text_to_synthesize': target_text_raw, 'target_language_code': self._get_cosyvoice_lang_code(target_lang)}
                
                response = requests.post(f"{COSYVOICE_API_URL}/generate-speech/", files=files, data=data, timeout=3600)

            if response.status_code != 200:
                raise RuntimeError(f"CosyVoice API failed: {response.status_code} - {response.text[:200]}")
                
            generated_audio_path = temp_dir / "cosy_output.wav"
            with open(generated_audio_path, "wb") as f:
                f.write(response.content)

            # --- Step 4: Temporal Mapping (CPU-based, low memory) ---
            adjusted_audio_path = self._apply_natural_temporal_mapping(generated_audio_path, word_timestamps or [], audio_numpy_16k, original_video_path, temp_dir, process_id_short)
            
            y_final, sr = librosa.load(str(adjusted_audio_path), sr=16000, mono=True)
            output_tensor = torch.from_numpy(y_final).unsqueeze(0)

            return {"audio": output_tensor, "transcripts": {"source": source_text_raw, "target": target_text_raw}}
    
    def is_language_supported(self, lang_code_app: str) -> bool:
        cosy_lang_code = self._get_cosyvoice_lang_code(lang_code_app)
        return cosy_lang_code in self.simple_lang_code_map_cosy.values()

    def get_supported_languages(self) -> Dict[str, str]:
        return {app_code: name for app_code, name in self.display_language_names.items()}

    def cleanup(self):
        logger.info("Cleaning CascadedBackend (CosyVoice API) resources...")
        if hasattr(self, 'asr_model'): del self.asr_model; self.asr_model = None; logger.debug("Whisper ASR model unloaded.")
        if hasattr(self, 'translator_model'): del self.translator_model; self.translator_model = None; logger.debug("NLLB Translator model unloaded.")
        if hasattr(self, 'translator_tokenizer'): del self.translator_tokenizer; self.translator_tokenizer = None; logger.debug("NLLB Tokenizer unloaded.")
        
        # ADD THESE LINES:
        if hasattr(self, 'visual_temporal_mapper'): 
            self.visual_temporal_mapper.cleanup()
            logger.debug("Visual temporal mapper cleaned up.")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import gc; gc.collect()
        logger.info("CascadedBackend (CosyVoice API) resources cleaned.")