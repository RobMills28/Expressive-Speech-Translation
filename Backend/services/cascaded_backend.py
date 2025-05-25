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
import json 

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None 

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().hasHandlers(): 
    import sys
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False 
    logger.info("CascadedBackend logger setup: No root handlers, added console handler.")
elif not logger.handlers and logging.getLogger().hasHandlers():
    logger.info("CascadedBackend logger setup: Will use root handlers.")

try:
    import whisper
    WHISPER_AVAILABLE_FLAG = True 
    logger.info("Whisper library FOUND for CascadedBackend.")
except ImportError:
    logger.warning("Whisper library NOT FOUND for CascadedBackend. Using HF pipeline.")
    WHISPER_AVAILABLE_FLAG = False
    whisper = None

OPENVOICE_API_URL = "http://localhost:8000"
OPENVOICE_API_AVAILABLE = False 

def check_openvoice_api():
    global OPENVOICE_API_AVAILABLE
    try:
        status_url = f"{OPENVOICE_API_URL}/status"
        logger.debug(f"Checking OpenVoice API: {status_url}")
        response = requests.get(status_url, timeout=3) 
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("tone_converter_model_loaded") and data.get("default_source_se_loaded"):
                    logger.info(f"OpenVoice API available, models loaded: {data.get('message', '')}")
                    OPENVOICE_API_AVAILABLE = True; return True
                else:
                    logger.warning(f"OpenVoice API up ({response.status_code}), but models NOT loaded: {data}")
                    OPENVOICE_API_AVAILABLE = False; return False
            except json.JSONDecodeError:
                logger.warning(f"OpenVoice API /status 200 but not valid JSON: {response.text[:200]}")
                OPENVOICE_API_AVAILABLE = False; return False
        elif response.status_code == 503:
             logger.warning(f"OpenVoice API /status 503: Service unavailable. Detail: {response.text[:200]}")
             OPENVOICE_API_AVAILABLE = False; return False
        else:
            logger.warning(f"OpenVoice API status check failed HTTP {response.status_code}. Detail: {response.text[:200]}")
            OPENVOICE_API_AVAILABLE = False; return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"OpenVoice API unavailable (RequestException): {type(e).__name__} - {e}")
        OPENVOICE_API_AVAILABLE = False; return False
    except Exception as e_gen: 
        logger.warning(f"OpenVoice API status check unexpected error: {type(e_gen).__name__} - {e_gen}")
        OPENVOICE_API_AVAILABLE = False; return False

check_openvoice_api() 
    
from .translation_strategy import TranslationBackend
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline

class CascadedBackend(TranslationBackend):
    def __init__(self, device=None, use_voice_cloning=True):
        logger.info(f"Initializing CascadedBackend. Device: {device}, Voice Cloning: {use_voice_cloning}")
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"CascadedBackend using ML device: {self.device}")
        self.initialized = False
        self.use_voice_cloning_config = use_voice_cloning 
        
        self.polly_voice_map = {'eng':{'VoiceId':'Joanna','Engine':'neural','Region':'us-east-1'},'deu':{'VoiceId':'Vicki','Engine':'neural','Region':'eu-central-1'},'spa':{'VoiceId':'Lucia','Engine':'neural','Region':'eu-west-1'},'fra':{'VoiceId':'Lea','Engine':'neural','Region':'eu-west-1'},'ita':{'VoiceId':'Bianca','Engine':'neural','Region':'eu-central-1'},'jpn':{'VoiceId':'Kazuha','Engine':'neural','Region':'ap-northeast-1'},'kor':{'VoiceId':'Seoyeon','Engine':'neural','Region':'ap-northeast-2'},'por':{'VoiceId':'Camila','Engine':'neural','Region':'sa-east-1'},'rus':{'VoiceId':'Tatyana','Engine':'standard','Region':'eu-central-1'},'cmn':{'VoiceId':'Zhiyu','Engine':'neural','Region':'ap-northeast-1'},'ara':{'VoiceId':'Zeina','Engine':'standard','Region':'me-south-1'},'hin':{'VoiceId':'Kajal','Engine':'neural','Region':'ap-south-1'},'nld':{'VoiceId':'Laura','Engine':'neural','Region':'eu-west-1'},'pol':{'VoiceId':'Ewa','Engine':'neural','Region':'eu-central-1'},'tur':{'VoiceId':'Filiz','Engine':'standard','Region':'eu-central-1'},'ukr':{'VoiceId':'Tatyana','Engine':'standard','Region':'eu-central-1'}}
        self.simple_lang_code_map = {'eng':'en','fra':'fr','spa':'es','deu':'de','ita':'it','por':'pt','cmn':'zh-cn','jpn':'ja','kor':'ko','ara':'ar','hin':'hi','nld':'nl','rus':'ru','pol':'pl','tur':'tr','ukr':'uk'}
        self.display_language_names = {'eng':'English (Joanna)','fra':'French (Lea)','spa':'Spanish (Lucia)','deu':'German (Vicki)','ita':'Italian (Bianca)','jpn':'Japanese (Kazuha)','kor':'Korean (Seoyeon)','por':'Portuguese (Camila)','rus':'Russian (Tatyana)','cmn':'Chinese (Zhiyu)','ara':'Arabic (Zeina)','hin':'Hindi (Kajal)','nld':'Dutch (Laura)','pol':'Polish (Ewa)','tur':'Turkish (Filiz)','ukr':'Ukrainian (Fallback)'}
        self.asr_model = None; self.asr_pipeline = None; self.translator_model = None; self.translator_tokenizer = None; self.polly_clients_cache = {} 
        logger.info(f"CascadedBackend __init__ flags: Whisper={WHISPER_AVAILABLE_FLAG}, OpenVoiceAPI={OPENVOICE_API_AVAILABLE}, PollySDK={BOTO3_AVAILABLE}")

    def _get_polly_client(self, region_name: str):
        if not BOTO3_AVAILABLE: logger.warning("Boto3 SDK not available."); return None
        if region_name in self.polly_clients_cache: return self.polly_clients_cache[region_name]
        try:
            logger.info(f"Creating Polly client for region: {region_name}"); client = boto3.client('polly', region_name=region_name)
            self.polly_clients_cache[region_name] = client; logger.info(f"Polly client for {region_name} created."); return client
        except Exception as e: logger.error(f"Failed to create Polly client for {region_name}: {e}", exc_info=True); return None

    def initialize(self):
        if self.initialized: logger.info("CascadedBackend already initialized."); return
        try:
            logger.info(f"Initializing CascadedBackend components on device: {self.device}"); start_time = time.time()
            if WHISPER_AVAILABLE_FLAG and whisper: self.asr_model = whisper.load_model("medium", device=self.device); logger.info("Whisper ASR loaded ('medium').")
            else: self.asr_pipeline = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if self.device.type == 'cuda' else -1); logger.info("Transformers ASR pipeline loaded ('openai/whisper-base').")
            self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            if self.device.type == 'cuda' and torch.cuda.is_available(): self.translator_model = self.translator_model.to(self.device)
            logger.info("NLLB translation model and tokenizer loaded.")
            if BOTO3_AVAILABLE: self._get_polly_client(os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            self.initialized = True; logger.info(f"CascadedBackend initialized successfully in {time.time() - start_time:.2f}s")
        except Exception as e: logger.error(f"Failed to initialize CascadedBackend: {e}", exc_info=True); self.initialized = False; raise 
    
    def _convert_to_nllb_code(self, lang_code_app: str) -> str:
        mapping = {'eng':'eng_Latn','fra':'fra_Latn','spa':'spa_Latn','deu':'deu_Latn','ita':'ita_Latn','por':'por_Latn','rus':'rus_Cyrl','cmn':'zho_Hans','jpn':'jpn_Jpan','kor':'kor_Hang','ara':'ara_Arab','hin':'hin_Deva','nld':'nld_Latn','pol':'pol_Latn','tur':'tur_Latn','ukr':'ukr_Cyrl','ces':'ces_Latn','hun':'hun_Latn'} 
        return mapping.get(lang_code_app.lower(), 'eng_Latn')

    def _get_simple_lang_code(self, lang_code_app: str) -> str: 
        return self.simple_lang_code_map.get(lang_code_app.lower(), 'en')

    def _generate_base_tts_audio_with_polly(self, text: str, lang_code_app: str, temp_audio_dir: Path, use_ssml: bool = False) -> Optional[Path]:
        polly_voice_config = self.polly_voice_map.get(lang_code_app);
        if not polly_voice_config: logger.warning(f"No Polly voice for '{lang_code_app}'."); return None
        polly_client = self._get_polly_client(polly_voice_config['Region']);
        if not polly_client: logger.warning(f"Polly client for '{polly_voice_config['Region']}' unavailable."); return None
        logger.debug(f"Polly TTS for '{lang_code_app}', VoiceId '{polly_voice_config['VoiceId']}'. SSML: {use_ssml}")
        mp3_path = temp_audio_dir / f"base_polly_{lang_code_app}.mp3"; wav_path = temp_audio_dir / f"base_polly_{lang_code_app}_16k.wav"
        try:
            req_params = {'Text': text, 'OutputFormat': 'mp3', 'VoiceId': polly_voice_config['VoiceId'], 'Engine': polly_voice_config['Engine']}
            if use_ssml: req_params['TextType'] = 'ssml';
            if use_ssml and not text.strip().lower().startswith("<speak>"): req_params['Text'] = f"<speak>{text}</speak>"
            resp = polly_client.synthesize_speech(**req_params)
            if 'AudioStream' in resp:
                with open(mp3_path, 'wb') as f: f.write(resp['AudioStream'].read())
                if not mp3_path.exists() or mp3_path.stat().st_size < 100: logger.error(f"Polly MP3 small/missing '{lang_code_app}'."); return None
                sound = AudioSegment.from_mp3(str(mp3_path)).set_frame_rate(16000).set_channels(1); sound.export(str(wav_path), format="wav")
                if not wav_path.exists() or wav_path.stat().st_size < 1000: logger.error(f"Polly WAV conversion failed '{lang_code_app}'."); return None
                logger.info(f"Polly 16kHz WAV OK for '{lang_code_app}': {wav_path.name} (Size: {wav_path.stat().st_size}b)")
                try: os.remove(mp3_path)
                except OSError: pass
                return wav_path
            else: logger.error(f"Polly no AudioStream '{lang_code_app}'. Resp: {resp}"); return None
        except Exception as e: logger.error(f"Polly TTS failed for '{lang_code_app}': {e}", exc_info=True); return None

    def _generate_base_tts_audio_with_gtts(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        logger.warning(f"Fallback gTTS for lang '{lang_code_app}'. Text: '{text[:70]}...'")
        try: from gtts import gTTS 
        except ImportError: logger.error("gTTS not installed."); return None
        gtts_code = self._get_simple_lang_code(lang_code_app); mp3_path = temp_audio_dir / f"base_gtts_{lang_code_app}_fb.mp3"
        wav_path = temp_audio_dir / f"base_gtts_{lang_code_app}_16k_fb.wav"
        try:
            tts = gTTS(text=text, lang=gtts_code, slow=False); tts.save(str(mp3_path))
            if not mp3_path.exists() or mp3_path.stat().st_size == 0: logger.error(f"gTTS MP3 failed '{lang_code_app}'."); return None
            sound = AudioSegment.from_mp3(str(mp3_path)).set_frame_rate(16000).set_channels(1); sound.export(str(wav_path), format="wav")
            if not wav_path.exists() or wav_path.stat().st_size < 1000: logger.error(f"gTTS WAV conversion failed '{lang_code_app}'."); return None
            logger.info(f"gTTS WAV for '{lang_code_app}': {wav_path.name}");
            try: os.remove(mp3_path)
            except OSError: pass
            return wav_path
        except Exception as e: logger.error(f"gTTS fallback failed '{lang_code_app}': {e}", exc_info=True); return None

    def _generate_base_tts_audio(self, text: str, lang_code_app: str, temp_audio_dir: Path) -> Optional[Path]:
        logger.debug(f"[_generate_base_tts_audio] Request for lang '{lang_code_app}', text: '{text[:70]}...'")
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map:
            polly_path = self._generate_base_tts_audio_with_polly(text, lang_code_app, temp_audio_dir)
            if polly_path and polly_path.exists(): return polly_path
            logger.warning(f"Polly failed for '{lang_code_app}'. Trying gTTS...")
        else: logger.info(f"Polly not for '{lang_code_app}'. Trying gTTS.")
        gtts_path = self._generate_base_tts_audio_with_gtts(text, lang_code_app, temp_audio_dir)
        if gtts_path and gtts_path.exists(): return gtts_path
        logger.error(f"All TTS failed for lang '{lang_code_app}'."); return None
        
    def _clone_voice_with_api(self, ref_voice_audio_path: str, target_content_audio_path: str, output_cloned_audio_path: str, req_id_short: str = "clone") -> bool:
        logger.info(f"[{req_id_short}] _clone_voice_with_api: Ref='{Path(ref_voice_audio_path).name}', Content='{Path(target_content_audio_path).name}'")
        
        if not check_openvoice_api(): 
            logger.warning(f"[{req_id_short}] OpenVoice API NOT available at time of cloning. Skipping."); return False
        
        if not Path(ref_voice_audio_path).exists() or Path(ref_voice_audio_path).stat().st_size < 1000:
            logger.error(f"[{req_id_short}] Reference voice audio for cloning MISSING/small: {ref_voice_audio_path}"); return False
        if not Path(target_content_audio_path).exists() or Path(target_content_audio_path).stat().st_size < 1000:
            logger.error(f"[{req_id_short}] Target content audio (TTS) for cloning MISSING/small: {target_content_audio_path}"); return False
        
        try:
            with open(ref_voice_audio_path, "rb") as f_ref, open(target_content_audio_path, "rb") as f_target_content:
                # --- KEYS MUST MATCH THE PYTHON PARAMETER NAMES IN openvoice_api.py (after aliases were removed there) ---
                files_payload = {
                    "reference_audio_file": (Path(ref_voice_audio_path).name, f_ref, "audio/wav"),
                    "content_audio_file": (Path(target_content_audio_path).name, f_target_content, "audio/wav")
                }
                logger.debug(f"[{req_id_short}] Sending files to OpenVoice API /clone-voice. Keys: {list(files_payload.keys())}")
                
                api_clone_url = f"{OPENVOICE_API_URL}/clone-voice"
                response = requests.post(api_clone_url, files=files_payload, timeout=180) 
            
            logger.info(f"[{req_id_short}] OpenVoice API /clone-voice response status: {response.status_code}")
            
            if response.status_code == 200:
                with open(output_cloned_audio_path, "wb") as f_out: 
                    f_out.write(response.content)
                if Path(output_cloned_audio_path).exists() and Path(output_cloned_audio_path).stat().st_size > 1000:
                    logger.info(f"[{req_id_short}] OpenVoice cloning successful. Output saved to: {output_cloned_audio_path}")
                    return True
                else:
                    size_info = Path(output_cloned_audio_path).stat().st_size if Path(output_cloned_audio_path).exists() else "DOES NOT EXIST"
                    logger.error(f"[{req_id_short}] OpenVoice API OK, but output file problematic: {output_cloned_audio_path} (Size: {size_info})")
                    return False
            else: 
                error_detail = f"OpenVoice API Error {response.status_code}"
                try:
                    error_data = response.json(); error_detail += f" - Detail: {error_data.get('detail', response.text[:200])}"
                except json.JSONDecodeError: error_detail += f" - Response: {response.text[:200]}" if response.text else " (No error text)"
                logger.error(f"[{req_id_short}] {error_detail}")
                return False
        except requests.exceptions.RequestException as e_req:
            logger.error(f"[{req_id_short}] RequestException during OpenVoice API call: {e_req}", exc_info=True); return False
        except Exception as e: 
            logger.error(f"[{req_id_short}] General exception in _clone_voice_with_api: {e}", exc_info=True); return False
        
    def translate_speech(self, audio_tensor: torch.Tensor, source_lang: str = "eng", target_lang: str = "fra") -> Dict[str, Any]:
        process_id = str(time.time())[-6:] 
        logger.info(f"[{process_id}] CascadedBackend.translate_speech CALLED. App Source: '{source_lang}', App Target: '{target_lang}'")
        if not self.initialized: 
            logger.info(f"[{process_id}] Backend not initialized, initializing now...")
            self.initialize()
            if not self.initialized: 
                 logger.error(f"[{process_id}] CRITICAL: Backend failed to initialize.")
                 return {"audio": torch.zeros((1, 1600), dtype=torch.float32), "transcripts": {"source": "ASR_FAILED_BACKEND_INIT", "target": "TRANSLATION_FAILED_BACKEND_INIT"}}

        start_time_translate_speech = time.time()
        with tempfile.TemporaryDirectory(prefix=f"cascaded_s2st_{process_id}_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            logger.info(f"[{process_id}] Using temp directory: {temp_dir}")
            source_text, target_text = "ASR_FAILED", "TRANSLATION_FAILED"; output_tensor = torch.zeros((1, 16000), dtype=torch.float32)
            try:
                asr_start_time = time.time(); audio_numpy = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
                if np.abs(audio_numpy).max() < 1e-5: source_text = "" ; logger.info(f"[{process_id}] Input audio near silent.")
                elif WHISPER_AVAILABLE_FLAG and self.asr_model:
                    lang_hint = self._get_simple_lang_code(source_lang); 
                    asr_result = self.asr_model.transcribe(audio_numpy, language=lang_hint if lang_hint != 'auto' else None, task="transcribe", fp16=(self.device.type == 'cuda'))
                    source_text = asr_result["text"]; logger.info(f"[{process_id}] Whisper ASR ({asr_result.get('language','unk')}): '{source_text[:70]}...' ({(time.time()-asr_start_time):.2f}s)")
                elif self.asr_pipeline: 
                    temp_asr_path = temp_dir / f"asr_in_{process_id}.wav"; sf.write(str(temp_asr_path), audio_numpy, 16000)
                    source_text = self.asr_pipeline(str(temp_asr_path))["text"]; logger.info(f"[{process_id}] Transformers ASR: '{source_text[:70]}...' ({(time.time()-asr_start_time):.2f}s)")
                    try: os.remove(temp_asr_path)
                    except: pass
                else: logger.error(f"[{process_id}] No ASR model.")
                
                translation_start_time = time.time()
                if not source_text or source_text == "ASR_FAILED": target_text = "" ; logger.info(f"[{process_id}] Source text empty. Translation skipped.")
                else:
                    src_code, tgt_code = self._convert_to_nllb_code(source_lang), self._convert_to_nllb_code(target_lang)
                    logger.info(f"[{process_id}] Translating NLLB '{src_code}' to '{tgt_code}'.")
                    self.translator_tokenizer.src_lang = src_code 
                    input_ids = self.translator_tokenizer(source_text, return_tensors="pt", padding=True).input_ids.to(self.device)
                    
                    forced_bos_token_id = None
                    # --- ROBUST NLLB BOS TOKEN ID RETRIEVAL ---
                    try:
                        if hasattr(self.translator_tokenizer, 'get_lang_id'): 
                            forced_bos_token_id = self.translator_tokenizer.get_lang_id(tgt_code)
                            logger.debug(f"[{process_id}] NLLB BOS for {tgt_code} via get_lang_id: {forced_bos_token_id}")
                        elif hasattr(self.translator_tokenizer, 'convert_tokens_to_ids'): 
                            forced_bos_token_id = self.translator_tokenizer.convert_tokens_to_ids(tgt_code)
                            logger.debug(f"[{process_id}] NLLB BOS for {tgt_code} via convert_tokens_to_ids: {forced_bos_token_id}")
                            if forced_bos_token_id == self.translator_tokenizer.unk_token_id:
                                logger.warning(f"[{process_id}] NLLB token for '{tgt_code}' was UNK. Trying alternate forms.")
                                # Try common variations if direct code is UNK
                                for prefix in ["", "<s> ", "<2", "[", "__{}__".format(tgt_code.split('_')[0])]: # Common prefixes/suffixes
                                    token_to_try = f"{prefix}{tgt_code}".strip() if prefix else tgt_code # Avoid double spaces
                                    alt_bos_id = self.translator_tokenizer.convert_tokens_to_ids(token_to_try)
                                    if alt_bos_id != self.translator_tokenizer.unk_token_id:
                                        forced_bos_token_id = alt_bos_id
                                        logger.info(f"[{process_id}] Found NLLB BOS for '{tgt_code}' as token '{token_to_try}': {forced_bos_token_id}")
                                        break 
                                if forced_bos_token_id == self.translator_tokenizer.unk_token_id: # If still UNK
                                    logger.error(f"[{process_id}] NLLB BOS token ID for '{tgt_code}' still UNK after trying common patterns.")
                                    forced_bos_token_id = None 
                        else: 
                            logger.error(f"[{process_id}] NLLB tokenizer does not have expected methods (get_lang_id or convert_tokens_to_ids).")
                            forced_bos_token_id = None
                    except Exception as e_bos:
                        logger.error(f"[{process_id}] Exception getting NLLB BOS token for {tgt_code}: {e_bos}", exc_info=True)
                        forced_bos_token_id = None
                    # --- END OF ROBUST NLLB BOS TOKEN ID RETRIEVAL ---
                    
                    if forced_bos_token_id: logger.info(f"[{process_id}] Using BOS ID: {forced_bos_token_id} for {tgt_code}")
                    else: logger.warning(f"[{process_id}] No valid forced_bos_token_id for {tgt_code}. NLLB might default.")

                    gen_kwargs = {"input_ids": input_ids, "max_length": max(256, len(input_ids[0]) * 3 + 50), "num_beams": 5, "length_penalty": 1.0, "early_stopping": True }
                    if forced_bos_token_id is not None: gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
                    
                    with torch.no_grad(): translated_tokens = self.translator_model.generate(**gen_kwargs)
                    target_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                logger.info(f"[{process_id}] NLLB Translation: '{target_text[:70]}...' ({(time.time()-translation_start_time):.2f}s)")
                
                tts_start_time = time.time(); base_tts_path = None
                if not target_text.strip(): logger.warning(f"[{process_id}] Target text empty. TTS silent.")
                else: base_tts_path = self._generate_base_tts_audio(target_text, target_lang, temp_dir)
                if base_tts_path and base_tts_path.exists():
                    y_base, _ = librosa.load(str(base_tts_path), sr=16000, mono=True) 
                    output_tensor = torch.from_numpy(y_base.astype(np.float32)).unsqueeze(0); logger.info(f"[{process_id}] Base TTS: {base_tts_path.name} ({(time.time()-tts_start_time):.2f}s)")
                else: logger.error(f"[{process_id}] Base TTS failed. Using silence. ({(time.time()-tts_start_time):.2f}s)")

                api_avail = check_openvoice_api() 
                should_clone = self.use_voice_cloning_config and api_avail and base_tts_path and base_tts_path.exists()
                logger.info(f"[{process_id}] OV clone decision: Conf={self.use_voice_cloning_config}, API={api_avail}, TTS_OK={base_tts_path and base_tts_path.exists()} -> Clone={should_clone}")
                if should_clone:
                    clone_t_start = time.time(); ref_path = temp_dir / f"orig_ref_{process_id}.wav"; sf.write(str(ref_path), audio_numpy, 16000) 
                    cloned_path = temp_dir / f"cloned_final_{process_id}.wav" 
                    clone_ok = self._clone_voice_with_api(str(ref_path), str(base_tts_path), str(cloned_path), req_id_short=process_id) 
                    if clone_ok and cloned_path.exists() and cloned_path.stat().st_size > 1000:
                        y_cloned, _ = librosa.load(str(cloned_path), sr=16000, mono=True)
                        output_tensor = torch.from_numpy(y_cloned.astype(np.float32)).unsqueeze(0); logger.info(f"[{process_id}] Using OV CLONED audio. ({(time.time()-clone_t_start):.2f}s)")
                    else: logger.warning(f"[{process_id}] OV cloning FAILED. Using base TTS. ({(time.time()-clone_t_start):.2f}s)")
                else: logger.info(f"[{process_id}] OV cloning SKIPPED.")
                logger.info(f"[{process_id}] Final audio shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
            except Exception as e: logger.error(f"[{process_id}] Error in translate_speech pipeline: {e}", exc_info=True); output_tensor = torch.zeros((1, 16000),dtype=torch.float32) 
            logger.info(f"[{process_id}] translate_speech completed in {time.time() - start_time_translate_speech:.2f}s")            
            return {"audio": output_tensor, "transcripts": {"source": source_text, "target": target_text}}

    def is_language_supported(self, lang_code_app: str) -> bool:
        if BOTO3_AVAILABLE and lang_code_app in self.polly_voice_map: return True
        if lang_code_app in self.simple_lang_code_map: return True 
        logger.warning(f"[is_language_supported] '{lang_code_app}' not supported.")
        return False
    
    def get_supported_languages(self) -> Dict[str, str]:
        supported = {}
        if BOTO3_AVAILABLE:
            for code, conf in self.polly_voice_map.items(): supported[code] = self.display_language_names.get(code, f"{code.upper()} (Polly: {conf['VoiceId']})")
        for code_gtts in self.simple_lang_code_map.keys():
            if code_gtts not in supported: supported[code_gtts] = self.display_language_names.get(code_gtts, code_gtts.upper()) + " (gTTS Fallback)"
        if not supported: logger.warning("[get_supported_languages] No Polly, listing gTTS only."); return {k: self.display_language_names.get(k, k.upper()) + " (gTTS)" for k in self.simple_lang_code_map.keys()}
        logger.debug(f"[get_supported_languages] Returning: {supported}")
        return supported

    def cleanup(self):
        logger.info("Cleaning CascadedBackend resources...");
        del self.asr_model; self.asr_model = None; del self.asr_pipeline; self.asr_pipeline = None
        del self.translator_model; self.translator_model = None; del self.translator_tokenizer; self.translator_tokenizer = None
        self.polly_clients_cache.clear()
        if self.device.type == 'cuda' and torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info("CascadedBackend resources cleaned.")