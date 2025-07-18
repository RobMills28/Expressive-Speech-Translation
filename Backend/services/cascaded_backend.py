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

from .temporal_mapper import TemporalMapper
from .translation_strategy import TranslationBackend
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        self.initialized = False
        self.asr_model = None
        self.translator_model = None
        self.translator_tokenizer = None

        self.simple_lang_code_map_cosy = {
            'eng': 'en', 'spa': 'es', 'fra': 'fr', 'deu': 'de', 'ita': 'it', 'por': 'pt', 'pol': 'pl',
            'tur': 'tr', 'rus': 'ru', 'nld': 'nl', 'ces': 'cs', 'ara': 'ar', 'cmn': 'zh',
            'jpn': 'ja', 'hun': 'hu', 'kor': 'ko', 'hin': 'hi'
        }
        self.display_language_names = {k: name.capitalize() for k, name in self.simple_lang_code_map_cosy.items()}
        self.display_language_names['cmn'] = 'Chinese (Mandarin)'

        if SAVE_DEBUG_AUDIO_FILES: logger.info("SAVE_DEBUG_AUDIO_FILES is enabled.")

    def initialize(self):
        if self.initialized: return
        logger.info(f"Initializing CascadedBackend components on device: {self.device}")
        start_time = time.time()
        try:
            try:
                import whisper
                self.asr_model = whisper.load_model("medium", device=self.device)
                logger.info("Whisper ASR loaded ('medium').")
            except ImportError:
                logger.error("Whisper library not found. ASR will not function.")
                raise

            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(self.device)
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            logger.info("NLLB translation model and tokenizer loaded.")

            api_healthy = self._check_cosyvoice_api_status()
            if not api_healthy:
                logger.error("CosyVoice API is not healthy. CascadedBackend initialization failed.")
                self.initialized = False
                return

            self.initialized = True
            logger.info(f"CascadedBackend (for CosyVoice API) initialized successfully in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize CascadedBackend (for CosyVoice API): {e}", exc_info=True)
            self.initialized = False; raise

    def _check_cosyvoice_api_status(self) -> bool:
        try:
            logger.info(f"Checking CosyVoice API status at {COSYVOICE_API_URL}/health")
            response = requests.get(f"{COSYVOICE_API_URL}/health", timeout=15)
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    api_status = health_data.get("status")
                    api_message = health_data.get("message", "")
                    if api_status == "healthy" and "CosyVoice model" in api_message and "loaded" in api_message:
                        logger.info(f"CosyVoice API is healthy: {api_message}")
                        return True
                    else:
                        logger.warning(f"CosyVoice API reported status '{api_status}' with message '{api_message}', which does not meet all health criteria. Full response: {health_data}")
                        return False
                except json.JSONDecodeError as e_json:
                    logger.error(f"CosyVoice API health response was not valid JSON: {e_json}. Status: {response.status_code}. Response text: {response.text[:500]}")
                    return False
            else:
                logger.warning(f"CosyVoice API status check HTTP request failed: {response.status_code} - {response.text[:200]}")
                return False
        except requests.exceptions.RequestException as e_req:
            logger.error(f"CosyVoice API request error for {COSYVOICE_API_URL}/health: {e_req}")
            return False
        except Exception as e_gen_status:
            logger.error(f"Unexpected error checking CosyVoice API status: {e_gen_status}", exc_info=True)
            return False

    def _convert_to_nllb_code(self, lang_code_app: str) -> str:
        mapping = {'eng':'eng_Latn','fra':'fra_Latn','spa':'spa_Latn','deu':'deu_Latn','ita':'ita_Latn','por':'por_Latn','rus':'rus_Cyrl','cmn':'zho_Hans','jpn':'jpn_Jpan','kor':'kor_Hang','ara':'ara_Arab','hin':'hin_Deva','nld':'nld_Latn','pol':'pol_Latn','tur':'tur_Latn','ukr':'ukr_Cyrl','ces':'ces_Latn','hun':'hun_Latn'}
        return mapping.get(lang_code_app.lower(), 'eng_Latn')

    def _get_cosyvoice_lang_code(self, lang_code_app: str) -> str:
        return self.simple_lang_code_map_cosy.get(lang_code_app.lower(), 'en')

    def _get_text_and_pauses_from_asr(self, audio_numpy: np.ndarray, source_lang: str, process_id_short: str) -> tuple[str, List[Dict[str, Any]], Optional[List[Dict[str,float]]]]:
        text_segments_list = []; pauses_info: List[Dict[str, Any]] = []; last_word_end_time = 0.0
        word_level_timestamps: Optional[List[Dict[str,float]]] = []

        if np.abs(audio_numpy).max() < 1e-5:
            return "", [], []
        if not self.asr_model:
            return "ASR_MODEL_UNAVAILABLE", [], []
        
        simple_source_lang = source_lang[:2] if source_lang else None
        logger.info(f"[{process_id_short}] Whisper ASR with word_timestamps=True, lang_hint='{simple_source_lang or 'auto'}'")
        import whisper
        asr_result = self.asr_model.transcribe(audio_numpy, language=simple_source_lang, task="transcribe", fp16=(self.device.type == 'cuda' if isinstance(self.device, torch.device) else self.device == 'cuda'), word_timestamps=True)

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
                                   temp_dir: Path, process_id_short: str) -> Path:
        """
        Apply natural temporal mapping to preserve speech characteristics
        """
        logger.info(f"[{process_id_short}] Applying natural temporal mapping...")
        
        try:
            # Load the generated audio
            generated_audio, sr = librosa.load(str(generated_audio_path), sr=16000, mono=True)
            
            # Extract timing profile from source audio
            source_profile = self.temporal_mapper.extract_timing_profile(source_audio_numpy, word_timestamps)
            
            logger.info(f"[{process_id_short}] Source profile: "
                    f"speaking_ratio={source_profile['speaking_ratio']:.2f}, "
                    f"avg_pause={source_profile['average_pause']:.2f}s, "
                    f"num_pauses={source_profile['num_pauses']}")
            
            # Apply temporal guidance
            adjusted_audio = self.temporal_mapper.apply_temporal_guidance(generated_audio, source_profile)
            
            # Save the adjusted audio
            adjusted_audio_path = temp_dir / f"temporally_mapped_{process_id_short}.wav"
            sf.write(str(adjusted_audio_path), adjusted_audio, 16000)
            
            # Log the results
            original_duration = len(generated_audio) / 16000
            adjusted_duration = len(adjusted_audio) / 16000
            
            logger.info(f"[{process_id_short}] Temporal mapping complete: "
                    f"original={original_duration:.2f}s, "
                    f"adjusted={adjusted_duration:.2f}s, "
                    f"change={adjusted_duration - original_duration:+.2f}s")
            
            return adjusted_audio_path
            
        except Exception as e:
            logger.error(f"[{process_id_short}] Error in temporal mapping: {e}", exc_info=True)
            return generated_audio_path

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
    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str = "eng", target_lang: str = "fra") -> Dict[str, Any]:
        process_id_short = str(time.time_ns())[-6:]
        logger.info(f"[{process_id_short}] CascadedBackend translate_speech with temporal mapping")
        
        if not self.initialized:
            raise RuntimeError("Backend not initialized. Please wait and try again.")
        if not self.asr_model or not self.translator_model:
            raise RuntimeError("Essential models (ASR/NLLB) are not loaded.")

        start_time_translate_speech = time.time()
        
        with tempfile.TemporaryDirectory(prefix=f"cosy_s2st_api_{process_id_short}_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            debug_audio_storage_dir = temp_dir / "debug_audio"
            debug_audio_storage_dir.mkdir(parents=True, exist_ok=True)

            source_text_raw = "ASR_FAILED"
            target_text_raw = "TRANSLATION_FAILED"
            output_tensor = torch.zeros((1, 16000), dtype=torch.float32)
            original_ref_for_eval_path = None
            cloned_audio_for_eval_path_16k = None

            try:
                # ASR Processing
                asr_start_time = time.time()
                audio_numpy_16k = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
                source_text_raw, original_pauses, word_timestamps = self._get_text_and_pauses_from_asr(
                    audio_numpy_16k, source_lang, process_id_short
                )
                
                logger.info(f"[{process_id_short}] ASR time: {(time.time()-asr_start_time):.2f}s.")
                
                if not source_text_raw.strip() or source_text_raw == "ASR_MODEL_UNAVAILABLE":
                    raise RuntimeError("ASR failed or produced empty text.")

                # Translation Processing
                translation_start_time = time.time()
                src_nllb_code = self._convert_to_nllb_code(source_lang)
                tgt_nllb_code = self._convert_to_nllb_code(target_lang)
                
                logger.info(f"[{process_id_short}] Translating NLLB '{src_nllb_code}' to '{tgt_nllb_code}'.")
                
                self.translator_tokenizer.src_lang = src_nllb_code
                input_ids = self.translator_tokenizer(source_text_raw, return_tensors="pt", padding=True).input_ids.to(self.device)
                forced_bos_token_id = self.translator_tokenizer.get_vocab().get(tgt_nllb_code)
                
                if forced_bos_token_id is None:
                    logger.warning(f"Could not get NLLB BOS token for {tgt_nllb_code}")
                
                gen_kwargs = {
                    "input_ids": input_ids,
                    "max_length": max(256, len(input_ids[0]) * 3 + 50),
                    "num_beams": 5,
                    "length_penalty": 1.0,
                    "early_stopping": True
                }
                
                if forced_bos_token_id is not None:
                    gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
                
                with torch.no_grad():
                    translated_tokens = self.translator_model.generate(**gen_kwargs)
                
                target_text_raw = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                logger.info(f"[{process_id_short}] Translation: '{target_text_raw[:70]}...' ({(time.time()-translation_start_time):.2f}s)")

                if not target_text_raw.strip():
                    raise RuntimeError("Translation result was empty.")

                # CosyVoice API Check
                if not self._check_cosyvoice_api_status():
                    raise RuntimeError("CosyVoice API is not healthy.")

                # CosyVoice Synthesis
                cosy_synthesis_start_time = time.time()
                cosy_target_lang_code = self._get_cosyvoice_lang_code(target_lang)
                reference_speaker_wav_for_api_path = self._get_reference_audio_for_cloning(
                    audio_numpy_16k, process_id_short, temp_dir, debug_audio_storage_dir
                )
                
                files = {
                    'reference_speaker_wav': (
                        reference_speaker_wav_for_api_path.name,
                        open(reference_speaker_wav_for_api_path, 'rb'),
                        'audio/wav'
                    )
                }
                
                data = {
                    'text_to_synthesize': target_text_raw,
                    'target_language_code': cosy_target_lang_code,
                    'style_prompt_text': ""
                }
                
                try:
                    response = requests.post(f"{COSYVOICE_API_URL}/generate-speech/", files=files, data=data, timeout=3600)
                    if response.status_code != 200:
                        raise RuntimeError(f"CosyVoice API failed: {response.status_code} - {response.text[:200]}")
                except requests.exceptions.RequestException as e_req:
                    raise RuntimeError(f"Could not connect to CosyVoice API: {e_req}")
                finally:
                    files['reference_speaker_wav'][1].close()

                logger.info(f"[{process_id_short}] CosyVoice API: {(time.time()-cosy_synthesis_start_time):.2f}s.")
                
                # Save CosyVoice output
                generated_audio_path_cosy_sr = temp_dir / f"cosyvoice_api_output_{process_id_short}.wav"
                with open(generated_audio_path_cosy_sr, 'wb') as f:
                    f.write(response.content)

                if not generated_audio_path_cosy_sr.exists() or generated_audio_path_cosy_sr.stat().st_size < 1000:
                    raise RuntimeError("CosyVoice API returned empty or missing audio file.")
                
                # Apply Natural Temporal Mapping (REPLACEMENT FOR OLD PAUSE ADJUSTMENT)
                temporal_mapping_start_time = time.time()
                adjusted_audio_path = self._apply_natural_temporal_mapping(
                    generated_audio_path_cosy_sr, word_timestamps or [], 
                    audio_numpy_16k, temp_dir, process_id_short
                )
                
                logger.info(f"[{process_id_short}] Temporal mapping: {(time.time()-temporal_mapping_start_time):.2f}s.")
                
                # Load final audio
                y_final_adjusted, sr_final_adjusted = librosa.load(str(adjusted_audio_path), sr=None, mono=True)
                
                if sr_final_adjusted != 16000:
                    y_final_16k = librosa.resample(y_final_adjusted, orig_sr=sr_final_adjusted, target_sr=16000)
                else:
                    y_final_16k = y_final_adjusted
                
                output_tensor = torch.from_numpy(y_final_16k.astype(np.float32)).unsqueeze(0)

                # Voice Similarity Check
                original_ref_for_eval_path = str(debug_audio_storage_dir / f"{process_id_short}_original_input_16k.wav")
                sf.write(original_ref_for_eval_path, audio_numpy_16k, 16000)

                cloned_audio_for_eval_path_16k = temp_dir / f"final_output_{process_id_short}_16k.wav"
                sf.write(str(cloned_audio_for_eval_path_16k), y_final_16k, 16000)
                
                if Path(original_ref_for_eval_path).exists() and Path(cloned_audio_for_eval_path_16k).exists():
                    try:
                        with open(original_ref_for_eval_path, 'rb') as f_orig, open(cloned_audio_for_eval_path_16k, 'rb') as f_cloned:
                            files_to_compare = {'original_audio': f_orig, 'cloned_audio': f_cloned}
                            sim_response = requests.post(f"{VOICE_SIMILARITY_API_URL}/compare-voices/", files=files_to_compare, timeout=60)
                        
                        if sim_response.status_code == 200:
                            similarity_score = sim_response.json().get('similarity_score', 'N/A')
                            logger.info(f"[{process_id_short}] Voice similarity: {similarity_score}")
                        else:
                            logger.warning(f"[{process_id_short}] Similarity API failed: {sim_response.status_code}")
                    except Exception as e_sim:
                        logger.warning(f"[{process_id_short}] Voice similarity check failed: {e_sim}")

                logger.info(f"[{process_id_short}] translate_speech completed successfully")
                return {
                    "audio": output_tensor,
                    "transcripts": {
                        "source": source_text_raw,
                        "target": target_text_raw
                    }
                }

            except Exception as e:
                logger.error(f"[{process_id_short}] Error in translate_speech: {e}", exc_info=True)
                raise e
    
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

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import gc; gc.collect()
        logger.info("CascadedBackend (CosyVoice API) resources cleaned.")