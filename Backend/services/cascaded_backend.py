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
            response = requests.get(f"{COSYVOICE_API_URL}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy" and health_data.get("message") == "CosyVoice model loaded.": # Check specific message
                    logger.info(f"CosyVoice API is healthy: {health_data.get('message')}")
                    return True
                else:
                    logger.warning(f"CosyVoice API reported unhealthy or model not loaded: {health_data}")
                    return False
            else:
                logger.warning(f"CosyVoice API status check failed: {response.status_code} - {response.text[:200]}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"CosyVoice API unreachable at {COSYVOICE_API_URL}/health: {e}")
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
            logger.info(f"[{process_id_short}] Input audio near silent for ASR.")
            return "", [], []

        if not self.asr_model:
            logger.error(f"[{process_id_short}] ASR model not loaded.")
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
                        word = word_info.get("word", "").strip()
                        start = word_info.get("start", 0.0)
                        end = word_info.get("end", 0.0)

                        word_level_timestamps.append({"word": word, "start": start, "end": end})

                        if current_global_word_index > 0 and start > last_word_end_time:
                            pause_duration = start - last_word_end_time
                            if pause_duration > 0.250:
                                pauses_info.append({
                                    "start": round(last_word_end_time, 3),
                                    "end": round(start, 3),
                                    "duration": round(pause_duration, 3),
                                    "insert_after_word_index": current_global_word_index -1
                                })
                        segment_text_parts.append(word)
                        last_word_end_time = end
                elif "text" in segment:
                    segment_text_parts.append(segment.get("text", "").strip())
                    logger.warning(f"[{process_id_short}] Segment without word timestamps: '{segment.get('text','')[:30]}...'. Pause accuracy might be affected for this part.")
                    last_word_end_time = segment.get("end", last_word_end_time)

                if segment_text_parts:
                    full_text_parts.append(" ".join(segment_text_parts))

        source_text = " ".join(full_text_parts).strip()
        if not source_text and asr_result.get("text", "").strip():
            source_text = asr_result.get("text", "").strip()
            pauses_info = []
            word_level_timestamps = []
            logger.warning(f"[{process_id_short}] No text from segments, using full ASR text. Pause info lost.")

        detected_lang = asr_result.get('language', 'unknown')
        logger.info(f"[{process_id_short}] Whisper ASR (Detected Lang: {detected_lang}): '{source_text[:70]}...'")
        logger.info(f"[{process_id_short}] Detected {len(pauses_info)} significant pauses.")
        if pauses_info: logger.debug(f"[{process_id_short}] Pauses: {json.dumps(pauses_info[:3], indent=2)}...")
        if word_level_timestamps: logger.debug(f"[{process_id_short}] Word timestamps count: {len(word_level_timestamps)}")

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
            logger.debug(f"Saved debug audio: {file_path} (Size: {file_path.stat().st_size if file_path.exists() else 'N/A'})")
        except Exception as e:
            logger.error(f"Failed to save debug audio {filename}: {e}", exc_info=True)

    def _log_audio_properties(self, audio_path: Path, label: str, process_id_short: str):
        if not audio_path or not audio_path.exists():
            logger.warning(f"[{process_id_short}] Cannot log properties for '{label}', file missing: {audio_path}")
            return
        try:
            y, sr_loaded = librosa.load(str(audio_path), sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr_loaded)
            peak_amp = np.abs(y).max()
            rms_amp = np.sqrt(np.mean(y**2))
            audio_segment = AudioSegment.from_file(str(audio_path))
            lufs = audio_segment.dBFS
            logger.info(f"[{process_id_short}] Properties for '{label}' ({audio_path.name}): Duration={duration:.2f}s, SR={sr_loaded}Hz, Peak={peak_amp:.4f}, RMS={rms_amp:.4f}, dBFS(LUFS_proxy)={lufs:.2f}")
        except Exception as e:
            logger.error(f"[{process_id_short}] Error logging properties for {label} ({audio_path.name}): {e}", exc_info=True)

    def _adjust_pauses_in_generated_audio(self, generated_audio_path: Path, original_pauses: List[Dict[str, Any]],
                                     original_total_speech_time: float, generated_total_speech_time: float,
                                     temp_dir: Path, process_id_short: str) -> Path:
        if not original_pauses:
            logger.info(f"[{process_id_short}] No original pauses to adjust in generated output.")
            return generated_audio_path

        logger.info(f"[{process_id_short}] Attempting to adjust pauses in generated output based on original.")
        logger.debug(f"Original total speech time: {original_total_speech_time:.2f}s, Generated total speech time: {generated_total_speech_time:.2f}s")

        try:
            generated_audio = AudioSegment.from_wav(generated_audio_path)
            output_frame_rate = generated_audio.frame_rate if generated_audio.frame_rate else 16000

            silence_thresh = generated_audio.dBFS - 16
            logger.debug(f"[{process_id_short}] Detecting silences in generated output with threshold: {silence_thresh:.2f} dBFS")

            generated_segments = silence.split_on_silence(
                generated_audio,
                min_silence_len=200,
                silence_thresh=silence_thresh,
                keep_silence=50
            )

            if not generated_segments:
                logger.warning(f"[{process_id_short}] Could not segment generated audio. Returning original generated output.")
                return generated_audio_path

            logger.info(f"[{process_id_short}] Generated audio split into {len(generated_segments)} speech segments.")

            final_audio_parts = []
            time_scaling_factor = generated_total_speech_time / original_total_speech_time if original_total_speech_time > 0 else 1.0

            num_pauses_to_insert = len(original_pauses)
            num_gaps_in_generated = len(generated_segments) - 1

            for i, segment in enumerate(generated_segments):
                final_audio_parts.append(segment)
                if i < num_gaps_in_generated:
                    if i < num_pauses_to_insert:
                        original_pause_duration_ms = int(original_pauses[i]['duration'] * 1000 * time_scaling_factor)
                        max_pause_ms = 3000
                        insert_pause_ms = min(original_pause_duration_ms, max_pause_ms)
                        if insert_pause_ms < 50: insert_pause_ms = 50

                        logger.debug(f"[{process_id_short}] Inserting adjusted pause of {insert_pause_ms}ms after generated segment {i+1}.")
                        final_audio_parts.append(AudioSegment.silent(duration=insert_pause_ms, frame_rate=output_frame_rate))
                    else:
                        logger.debug(f"[{process_id_short}] Adding default short pause after generated segment {i+1}.")
                        final_audio_parts.append(AudioSegment.silent(duration=200, frame_rate=output_frame_rate))

            adjusted_audio = sum(final_audio_parts, AudioSegment.empty())
            adjusted_audio_path = temp_dir / f"generated_adjusted_pauses_{process_id_short}.wav"
            adjusted_audio.export(adjusted_audio_path, format="wav")
            logger.info(f"[{process_id_short}] Exported generated audio with adjusted pauses: {adjusted_audio_path.name}")
            self._log_audio_properties(adjusted_audio_path, "GeneratedAudio_AdjustedPauses", process_id_short)
            return adjusted_audio_path

        except Exception as e:
            logger.error(f"[{process_id_short}] Error adjusting pauses in generated output: {e}", exc_info=True)
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
        sf.write(str(ref_audio_path), ref_audio_np, target_sr_for_ref)

        if debug_audio_storage_dir:
             self._save_debug_audio(ref_audio_np, f"{process_id_short}_ref_for_cosyAPI_sent_16k.wav", debug_audio_storage_dir, target_sr_for_ref)
        self._log_audio_properties(ref_audio_path, f"RefAudioForCosyVoiceAPI_16kHz", process_id_short)
        return ref_audio_path


    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str = "eng", target_lang: str = "fra") -> Dict[str, Any]:
        process_id_short = str(time.time_ns())[-6:]
        logger.info(f"[{process_id_short}] CascadedBackend (CosyVoice API).translate_speech CALLED. App Source: '{source_lang}', App Target: '{target_lang}'")
        if not self.initialized:
            logger.error(f"[{process_id_short}] Backend not initialized. Aborting.")
            return {"audio": torch.zeros((1,16000),dtype=torch.float32), "transcripts": {"source":"INIT_ERROR","target":"INIT_ERROR"}}
        if not self.asr_model or not self.translator_model :
             logger.error(f"[{process_id_short}] Essential models (ASR/NLLB) not initialized. Aborting.")
             return {"audio": torch.zeros((1,16000),dtype=torch.float32), "transcripts": {"source":"MODEL_INIT_FAIL","target":"MODEL_INIT_FAIL"}}

        start_time_translate_speech = time.time()
        with tempfile.TemporaryDirectory(prefix=f"cosy_s2st_api_{process_id_short}_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            debug_audio_storage_dir = temp_dir / "debug_audio"
            if SAVE_DEBUG_AUDIO_FILES: debug_audio_storage_dir.mkdir(parents=True, exist_ok=True)

            source_text_raw, original_pauses, word_timestamps = "ASR_FAILED", [], []
            target_text_raw = "TRANSLATION_FAILED"
            output_tensor = torch.zeros((1, 16000), dtype=torch.float32)
            original_ref_for_eval_path = None
            cloned_audio_for_eval_path_16k = None

            try:
                asr_start_time = time.time()
                audio_numpy_16k = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
                source_text_raw, original_pauses, word_timestamps = self._get_text_and_pauses_from_asr(audio_numpy_16k, source_lang, process_id_short)
                original_total_speech_time = sum(item['end'] - item['start'] for item in word_timestamps) if word_timestamps else librosa.get_duration(y=audio_numpy_16k, sr=16000)
                logger.info(f"[{process_id_short}] ASR time: {(time.time()-asr_start_time):.2f}s. Text: '{source_text_raw[:70]}...'. Pauses: {len(original_pauses)}")
                if SAVE_DEBUG_AUDIO_FILES:
                    sf.write(str(debug_audio_storage_dir / f"{process_id_short}_original_input_for_asr_16k.wav"), audio_numpy_16k, 16000)

                translation_start_time = time.time()
                if not source_text_raw.strip() or source_text_raw == "ASR_MODEL_UNAVAILABLE":
                    target_text_raw = ""
                else:
                    src_nllb_code = self._convert_to_nllb_code(source_lang)
                    tgt_nllb_code = self._convert_to_nllb_code(target_lang)
                    logger.info(f"[{process_id_short}] Translating NLLB '{src_nllb_code}' to '{tgt_nllb_code}'.")
                    self.translator_tokenizer.src_lang = src_nllb_code
                    input_ids = self.translator_tokenizer(source_text_raw, return_tensors="pt", padding=True).input_ids.to(self.device)

                    forced_bos_token_id = self.translator_tokenizer.lang_code_to_id.get(tgt_nllb_code)
                    if forced_bos_token_id is None:
                        logger.warning(f"Could not get NLLB BOS token for {tgt_nllb_code}, translation might be suboptimal.")

                    gen_kwargs = {"input_ids": input_ids, "max_length": max(256, len(input_ids[0]) * 3 + 50),
                                  "num_beams": 5, "length_penalty": 1.0, "early_stopping": True}
                    if forced_bos_token_id is not None:
                        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

                    with torch.no_grad(): translated_tokens = self.translator_model.generate(**gen_kwargs)
                    target_text_raw = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                logger.info(f"[{process_id_short}] NLLB Raw Translation: '{target_text_raw[:70]}...' ({(time.time()-translation_start_time):.2f}s)")

                cosy_synthesis_start_time = time.time()
                cosy_target_lang_code = self._get_cosyvoice_lang_code(target_lang)

                reference_speaker_wav_for_api_path = self._get_reference_audio_for_cloning(
                    audio_numpy_16k, process_id_short, temp_dir, debug_audio_storage_dir
                )

                generated_audio_path_cosy_sr = None

                if not target_text_raw.strip():
                    logger.warning(f"[{process_id_short}] Target text for CosyVoice API is empty. Using silent output tensor.")
                elif not self._check_cosyvoice_api_status():
                    logger.error(f"[{process_id_short}] CosyVoice API not healthy/available at time of synthesis. Using silent output tensor.")
                else:
                    logger.info(f"[{process_id_short}] Calling CosyVoice API. Target lang: '{cosy_target_lang_code}'. Ref: '{reference_speaker_wav_for_api_path.name}'. Text: '{target_text_raw[:70]}...'")

                    files = {'reference_speaker_wav': (reference_speaker_wav_for_api_path.name, open(reference_speaker_wav_for_api_path, 'rb'), 'audio/wav')}
                    data = {
                        'text_to_synthesize': target_text_raw,
                        'target_language_code': cosy_target_lang_code,
                        'style_prompt_text': ""
                    }

                    try:
                        response = requests.post(f"{COSYVOICE_API_URL}/generate-speech/", files=files, data=data, timeout=300)
                        files['reference_speaker_wav'][1].close()

                        if response.status_code == 200:
                            generated_audio_path_cosy_sr = temp_dir / f"cosyvoice_api_output_{process_id_short}.wav"
                            with open(generated_audio_path_cosy_sr, 'wb') as f:
                                f.write(response.content)
                            if not generated_audio_path_cosy_sr.exists() or generated_audio_path_cosy_sr.stat().st_size < 1000:
                                logger.error(f"[{process_id_short}] CosyVoice API returned 200 but output file is too small or missing.")
                                generated_audio_path_cosy_sr = None
                            else:
                                logger.info(f"[{process_id_short}] Received audio from CosyVoice API: {generated_audio_path_cosy_sr.name}")
                                self._log_audio_properties(generated_audio_path_cosy_sr, "CosyVoiceAPI_Output_NativeSR", process_id_short)
                        else:
                            logger.error(f"[{process_id_short}] CosyVoice API call failed: {response.status_code} - {response.text[:500]}")
                    except requests.exceptions.RequestException as e_req:
                        logger.error(f"[{process_id_short}] Request to CosyVoice API failed: {e_req}", exc_info=True)

                logger.info(f"[{process_id_short}] CosyVoice API interaction took: {(time.time()-cosy_synthesis_start_time):.2f}s.")

                if generated_audio_path_cosy_sr and generated_audio_path_cosy_sr.exists():
                    y_cosy_native_sr, sr_cosy_native = librosa.load(str(generated_audio_path_cosy_sr), sr=None, mono=True)
                    generated_total_speech_time = librosa.get_duration(y=y_cosy_native_sr, sr=sr_cosy_native)

                    adjusted_audio_path = self._adjust_pauses_in_generated_audio(
                        generated_audio_path_cosy_sr, original_pauses,
                        original_total_speech_time, generated_total_speech_time,
                        temp_dir, process_id_short
                    )

                    y_final_adjusted, sr_final_adjusted = librosa.load(str(adjusted_audio_path), sr=None, mono=True)
                    if sr_final_adjusted != 16000:
                        y_final_16k = librosa.resample(y_final_adjusted, orig_sr=sr_final_adjusted, target_sr=16000)
                    else:
                        y_final_16k = y_final_adjusted
                    output_tensor = torch.from_numpy(y_final_16k.astype(np.float32)).unsqueeze(0)

                    original_ref_for_eval_path = str(debug_audio_storage_dir / f"{process_id_short}_original_input_for_asr_16k.wav")

                    temp_cloned_for_eval_path = temp_dir / f"cosyvoice_cloned_for_eval_{process_id_short}_16k.wav"
                    sf.write(str(temp_cloned_for_eval_path), y_final_16k, 16000)
                    cloned_audio_for_eval_path_16k = str(temp_cloned_for_eval_path)
                    self._log_audio_properties(Path(cloned_audio_for_eval_path_16k), "CosyVoice_Final_16kHz_for_Eval", process_id_short)
                else:
                    logger.warning(f"[{process_id_short}] No valid audio received from CosyVoice API. Output tensor is silent 16kHz.")


                if original_ref_for_eval_path and Path(original_ref_for_eval_path).exists() and \
                   cloned_audio_for_eval_path_16k and Path(cloned_audio_for_eval_path_16k).exists():
                    try:
                        with open(original_ref_for_eval_path, 'rb') as f_orig, open(cloned_audio_for_eval_path_16k, 'rb') as f_cloned:
                            files_to_compare = {
                                'original_audio': (Path(original_ref_for_eval_path).name, f_orig, 'audio/wav'),
                                'cloned_audio': (Path(cloned_audio_for_eval_path_16k).name, f_cloned, 'audio/wav')
                            }
                            logger.info(f"[{process_id_short}] Calling voice similarity API with 16kHz files.")
                            api_response = requests.post(f"{VOICE_SIMILARITY_API_URL}/compare-voices/", files=files_to_compare, timeout=60)

                        if api_response.status_code == 200:
                            similarity_data = api_response.json()
                            similarity_score = similarity_data.get("similarity_score")
                            if similarity_score is not None: logger.info(f"[{process_id_short}] VOICE SIMILARITY SCORE (CosyVoice vs Original): {similarity_score:.4f}")
                            else: logger.warning(f"[{process_id_short}] Similarity API response missing score: {similarity_data}")
                        else: logger.error(f"[{process_id_short}] Voice similarity API call failed with status {api_response.status_code}: {api_response.text[:200]}")
                    except Exception as e_sim_eval:
                        logger.error(f"[{process_id_short}] Error during voice similarity evaluation: {e_sim_eval}", exc_info=True)
                else:
                    logger.warning(f"[{process_id_short}] Skipping similarity check due to missing audio files for evaluation.")

                logger.info(f"[{process_id_short}] Final audio tensor for output: shape: {output_tensor.shape}, dtype: {output_tensor.dtype}, SR assumed 16kHz")

            except Exception as e:
                logger.error(f"[{process_id_short}] Error in CosyVoice API translate_speech pipeline: {e}", exc_info=True)
                output_tensor = torch.zeros((1, 16000),dtype=torch.float32)
                source_text_raw = source_text_raw if source_text_raw != "ASR_FAILED" else "PIPELINE_ASR_ERROR"
                target_text_raw = "PIPELINE_ERROR"

            logger.info(f"[{process_id_short}] translate_speech (CosyVoice API) completed in {time.time() - start_time_translate_speech:.2f}s")
            return {"audio": output_tensor, "transcripts": {"source": source_text_raw, "target": target_text_raw}}

    def is_language_supported(self, lang_code_app: str) -> bool:
        cosy_lang_code = self._get_cosyvoice_lang_code(lang_code_app)
        return cosy_lang_code in self.simple_lang_code_map_cosy.values()

    def get_supported_languages(self) -> Dict[str, str]:
        return {app_code: name for app_code, name in self.display_language_names.items()}

    def cleanup(self):
        logger.info("Cleaning CascadedBackend (CosyVoice API) resources...")
        if hasattr(self, 'asr_model') and self.asr_model: del self.asr_model; self.asr_model = None; logger.debug("Whisper ASR model unloaded.")
        if hasattr(self, 'translator_model') and self.translator_model: del self.translator_model; self.translator_model = None; logger.debug("NLLB Translator model unloaded.")
        if hasattr(self, 'translator_tokenizer') and self.translator_tokenizer: del self.translator_tokenizer; self.translator_tokenizer = None; logger.debug("NLLB Tokenizer unloaded.")

        current_device_type = self.device.type if isinstance(self.device, torch.device) else self.device
        if current_device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache(); logger.debug("CUDA cache emptied.")

        import gc; gc.collect()
        logger.info("CascadedBackend (CosyVoice API) resources cleaned.")