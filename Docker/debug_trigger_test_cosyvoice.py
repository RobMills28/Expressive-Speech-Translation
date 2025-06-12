# THIS IS THE TEMPORARY DEBUG FILE: Magenta AI/Docker/debug_trigger_test_cosyvoice.py
# It will be COPIED into the Docker image to replace the original library file for testing.

print("DEBUG cosyvoice.py (debug_trigger_test_cosyvoice.py): TOP OF FILE") # Modified print

import os
print("DEBUG cosyvoice.py: Imported os")
import time
print("DEBUG cosyvoice.py: Imported time")
from typing import Generator
print("DEBUG cosyvoice.py: Imported Generator from typing")

# --- Reverted Test 1 ---
from omegaconf import OmegaConf # <<<< UNCOMMENTED
print("DEBUG cosyvoice.py: Imported OmegaConf from omegaconf") # <<<< UNCOMMENTED
# --- End Reverted Test 1 ---

# --- TEST 2: Commenting out hyperpyyaml ---
# from hyperpyyaml import load_hyperpyyaml # <<<< COMMENTED OUT
# print("DEBUG cosyvoice.py: Imported load_hyperpyyaml from hyperpyyaml")
print("DEBUG cosyvoice.py: SKIPPED HYPERPYYAML IMPORT FOR TEST 2") # <<<< ADDED THIS
# --- END TEST 2 ---

from modelscope import snapshot_download
print("DEBUG cosyvoice.py: Imported snapshot_download from modelscope")

import torch
print("DEBUG cosyvoice.py: Imported torch")
import torchaudio
print("DEBUG cosyvoice.py: Imported torchaudio")

from cosyvoice.cli.frontend import CosyVoiceFrontend
print("DEBUG cosyvoice.py: Imported CosyVoiceFrontend from cosyvoice.cli.frontend")
from cosyvoice.model import CosyVoiceModel, CosyVoice2Model
print("DEBUG cosyvoice.py: Imported CosyVoiceModel, CosyVoice2Model from cosyvoice.model")
from cosyvoice.utils.file_utils import load_wav
print("DEBUG cosyvoice.py: Imported load_wav from cosyvoice.utils.file_utils")
from cosyvoice.utils.misc_utils import get_spk_model_path, get_spk_type
print("DEBUG cosyvoice.py: Imported get_spk_model_path, get_spk_type from cosyvoice.utils.misc_utils")
import logging # Added missing import

# --- Dummy get_model_type ---
def get_model_type(config):
    print("DEBUG: dummy get_model_type called")
    if configs and 'llm' in configs and isinstance(configs['llm'], dict) and configs['llm'].get('type') == 'CosyVoice2LLM': # Example check
         return CosyVoice2Model
    return CosyVoiceModel # Default fallback
# --- End Dummy ---

class CosyVoice(object):
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        print(f"DEBUG CosyVoice.__init__: model_dir={model_dir}")
        self.instruct = True if "-Instruct" in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            print(f"DEBUG CosyVoice.__init__: model_dir not found, using snapshot_download for {model_dir}")
            model_dir = snapshot_download(model_dir)
            print(f"DEBUG CosyVoice.__init__: snapshot_download returned {model_dir}")
        self.model_dir = model_dir # Ensure self.model_dir is updated

        hyper_yam_path = '{}/cosyvoice.yaml'.format(self.model_dir)
        print(f"DEBUG CosyVoice.__init__: looking for hyper_yam_path={hyper_yam_path}")
        if not os.path.exists(hyper_yam_path):
            raise ValueError('{} not found!'.format(hyper_yam_path))
        
        configs = None
        name_error_occurred = False
        try:
            print(f"DEBUG CosyVoice.__init__: Attempting to load config via load_hyperpyyaml (will fail if commented out)")
            with open(hyper_yam_path, 'r') as f:
                configs = load_hyperpyyaml(f) # This will raise NameError if load_hyperpyyaml is not defined
            print("DEBUG CosyVoice.__init__: configs loaded via load_hyperpyyaml")
        except NameError as ne:
            if 'load_hyperpyyaml' in str(ne): 
                print(f"DEBUG CosyVoice.__init__: Caught NameError for commented 'load_hyperpyyaml': {ne}. THIS IS EXPECTED FOR THIS TEST.")
                name_error_occurred = True
                pass 
            else:
                raise 
        except Exception as e_conf:
            print(f"DEBUG CosyVoice.__init__: Error loading config: {e_conf}")
            raise

        if configs is None and not name_error_occurred:
             raise ValueError("Failed to load configs and load_hyperpyyaml was not the NameError culprit.")

        if model_dir.endswith('.pt'):
            print("DEBUG CosyVoice.__init__: model_dir ends with .pt branch")
            self.model, self.config, self.tokenizer = CosyVoiceFrontend.from_pretrained(self.model_dir, fp16=self.fp16)
        else:
            print("DEBUG CosyVoice.__init__: model_dir is a directory branch")
            self.config = configs
            expected_model_type = CosyVoiceModel
            if configs:
                assert get_model_type(configs) == expected_model_type, 'do not use {} for CosyVoice initialization! (Expected {}) Got {}'.format(self.model_dir, expected_model_type, get_model_type(configs) if configs else "None")
            else:
                print("DEBUG CosyVoice.__init__: Skipping model type assert as configs are None (due to debug).")

            print("DEBUG CosyVoice.__init__: Initializing CosyVoiceFrontend...")
            self.frontend = CosyVoiceFrontend(configs['get_tokenizer'] if configs else None,
                                              configs.get('campplus.onnx',None) if configs and 'campplus.onnx' in configs else '{}/campplus.onnx'.format(self.model_dir),
                                              configs.get('speech_tokenizer_v2.onnx', None) if configs and 'speech_tokenizer_v2.onnx' in configs else '{}/speech_tokenizer_v2.onnx'.format(self.model_dir),
                                              configs['allowed_special'] if configs else [])
            print("DEBUG CosyVoice.__init__: CosyVoiceFrontend initialized.")

            self.sample_rate = configs['sample_rate'] if configs else 22050
            if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
                logging.warning("no cuda device, set load_jit/load_trt/fp16 to False")
                load_jit, load_trt, fp16 = False, False, False
            
            print("DEBUG CosyVoice.__init__: Initializing CosyVoiceModel...")
            self.model = CosyVoiceModel(configs['llm'] if configs else None, configs['flow'] if configs else None, configs['hift'] if configs else None, fp16)
            print("DEBUG CosyVoice.__init__: CosyVoiceModel initialized. Loading state dicts...")
            self.model.load_state_dict(torch.load('{}/llm.pt'.format(self.model_dir), map_location='cpu'), strict=False)
            print("DEBUG CosyVoice.__init__: Loaded llm.pt")
            self.model.load_state_dict(torch.load('{}/flow.pt'.format(self.model_dir), map_location='cpu'), strict=False)
            print("DEBUG CosyVoice.__init__: Loaded flow.pt")
            self.model.load_state_dict(torch.load('{}/hift.pt'.format(self.model_dir), map_location='cpu'), strict=False)
            print("DEBUG CosyVoice.__init__: Loaded hift.pt. State dicts loaded.")

            if load_vllm: raise NotImplementedError("load_vllm not implemented")
            if load_jit:
                print("DEBUG CosyVoice.__init__: Loading JIT models...")
                self.model.load_jit('{}/llm.text_encoder.j'.format(self.model_dir), '{}/llm.flow_decoder.j'.format(self.model_dir),
                                    '{}/llm.final_proj.j'.format(self.model_dir), fp16 if self.fp16 is True else 'fp32')
                print("DEBUG CosyVoice.__init__: JIT models loaded.")
            if load_trt:
                print("DEBUG CosyVoice.__init__: Loading TRT models...")
                self.model.load_trt('{}/llm.decoder.estimator.j'.format(self.model_dir), '{}/flow.decoder.estimator.fp32.onnx'.format(self.model_dir),
                                    trt_concurrent, self.fp16)
                print("DEBUG CosyVoice.__init__: TRT models loaded.")
        print("DEBUG CosyVoice.__init__: __init__ finished.")

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.zero_shot_frontend(prompt_text, prompt_speech_16k, self.sample_rate, '')
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, os.path.join(self.model_dir, "spkinfo.pt"))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        model_input = self.frontend.sft_frontend(tts_text, spk_id, stream, text_frontend=text_frontend)
        for i in tts_text.split(): 
            logging.info('synthesis text {}'.format(i))
            start_time = time.time()
            for model_output in self.model.sft_model(model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
            start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        model_input = self.frontend.zero_shot_frontend(prompt_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
        logging.warning('synthesis text "{}", this may lead to bad performance. prompt_text "{}"'.format(tts_text if isinstance(tts_text, str) else "GENERATOR_INPUT", prompt_text))
        start_time = time.time()
        for model_output in self.model.tts_model(model_input, stream=stream, speed=speed): 
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
        start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        model_input = self.frontend.zero_shot_frontend('', prompt_speech_16k, self.sample_rate, zero_shot_spk_id) 
        logging.warning('synthesis text "{}"'.format(tts_text if isinstance(tts_text, str) else "GENERATOR_INPUT"))
        start_time = time.time()
        for model_output in self.model.tts_model(model_input, text=tts_text, text_frontend=text_frontend, 
                                                 normalize_text=True, 
                                                 split_text=True, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
        start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('do not support instruct inference'.format(self.model_dir))
        model_input = self.frontend.instruct(tts_text, spk_id, instruct_text, text_frontend=text_frontend)
        for i in tts_text.split(): 
            logging.info('synthesis text {}'.format(i))
            start_time = time.time()
            for model_output in self.model.instruct_tts(model_input, stream=stream, speed=speed): 
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
            start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.vc_frontend(source_speech_16k, prompt_speech_16k, self.sample_rate)
        logging.info('synthesis text {}'.format('')) 
        start_time = time.time()
        for model_output in self.model.vc_model(model_input, stream=stream, speed=speed): 
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
        start_time = time.time()


class CosyVoice2(CosyVoice):
    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        print(f"DEBUG CosyVoice2.__init__: model_dir={model_dir}")
        self.instruct = True if "-Instruct" in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            print(f"DEBUG CosyVoice2.__init__: model_dir not found, using snapshot_download for {model_dir}")
            model_dir = snapshot_download(model_dir)
            print(f"DEBUG CosyVoice2.__init__: snapshot_download returned {model_dir}")
        self.model_dir = model_dir

        hyper_yam_path = '{}/cosyvoice.yaml'.format(self.model_dir)
        print(f"DEBUG CosyVoice2.__init__: looking for hyper_yam_path={hyper_yam_path}")
        if not os.path.exists(hyper_yam_path):
            hyper_yam_path_v2_original = '{}/cosyvoice2.yaml'.format(self.model_dir)
            print(f"DEBUG CosyVoice2.__init__: {hyper_yam_path} not found, trying {hyper_yam_path_v2_original}")
            if os.path.exists(hyper_yam_path_v2_original):
                hyper_yam_path = hyper_yam_path_v2_original
            else:
                raise ValueError('Neither cosyvoice.yaml (renamed) nor cosyvoice2.yaml (original) found! Searched: {}, {}'.format(hyper_yam_path, hyper_yam_path_v2_original))
        
        configs = None
        name_error_occurred = False # Specific to this debug version
        try:
            print(f"DEBUG CosyVoice2.__init__: Attempting to load config via load_hyperpyyaml with overrides (will fail if commented out)")
            with open(hyper_yam_path, 'r') as f:
                configs = load_hyperpyyaml(f, overrides={'qwen_pretrained_path': os.path.join(self.model_dir, 'CosyVoice-BlankEN')})
            print("DEBUG CosyVoice2.__init__: configs loaded via load_hyperpyyaml with overrides")
        except NameError as ne:
             if 'load_hyperpyyaml' in str(ne): 
                print(f"DEBUG CosyVoice2.__init__: Caught NameError for commented 'load_hyperpyyaml': {ne}. THIS IS EXPECTED FOR THIS TEST.")
                name_error_occurred = True
                pass
             else:
                raise
        except Exception as e_conf:
            print(f"DEBUG CosyVoice2.__init__: Error loading config with overrides: {e_conf}")
            raise
        
        if configs is None and not name_error_occurred:
             raise ValueError("Failed to load configs (CosyVoice2) and load_hyperpyyaml was not the NameError culprit.")

        expected_model_type = CosyVoice2Model
        if configs:
            assert get_model_type(configs) == expected_model_type, 'do not use {} for CosyVoice2 initialization! (Expected {}) Got {}'.format(self.model_dir, expected_model_type, get_model_type(configs) if configs else "None")
        else:
            print("DEBUG CosyVoice2.__init__: Skipping model type assert as configs are None (due to debug).")

        print("DEBUG CosyVoice2.__init__: Initializing CosyVoiceFrontend...")
        self.frontend = CosyVoiceFrontend(configs['get_tokenizer'] if configs else None,
                                          configs.get('campplus.onnx', None) if configs and 'campplus.onnx' in configs else '{}/campplus.onnx'.format(self.model_dir),
                                          configs.get('speech_tokenizer_v2.onnx', None) if configs and 'speech_tokenizer_v2.onnx' in configs else '{}/speech_tokenizer_v2.onnx'.format(self.model_dir),
                                          configs['allowed_special'] if configs else [])
        print("DEBUG CosyVoice2.__init__: CosyVoiceFrontend initialized.")

        self.sample_rate = configs['sample_rate'] if configs else 22050
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            logging.warning("no cuda device, set load_jit/load_trt/fp16 to False")
            load_jit, load_trt, fp16 = False, False, False
        
        print("DEBUG CosyVoice2.__init__: Initializing CosyVoice2Model...")
        self.model = CosyVoice2Model(configs['llm'] if configs else None, configs['flow'] if configs else None, configs['hift'] if configs else None, fp16)
        print("DEBUG CosyVoice2.__init__: CosyVoice2Model initialized. Loading state dicts...")
        self.model.load_state_dict(torch.load('{}/llm.pt'.format(self.model_dir), map_location='cpu'), strict=False)
        print("DEBUG CosyVoice2.__init__: Loaded llm.pt")
        self.model.load_state_dict(torch.load('{}/flow.pt'.format(self.model_dir), map_location='cpu'), strict=False)
        print("DEBUG CosyVoice2.__init__: Loaded flow.pt")
        self.model.load_state_dict(torch.load('{}/hift.pt'.format(self.model_dir), map_location='cpu'), strict=False)
        print("DEBUG CosyVoice2.__init__: Loaded hift.pt. State dicts loaded.")

        if load_vllm:
            print("DEBUG CosyVoice2.__init__: Loading VLLM model...")
            self.model.load_vllm('{}/vllm'.format(self.model_dir))
            print("DEBUG CosyVoice2.__init__: VLLM model loaded.")

        if load_jit:
            print("DEBUG CosyVoice2.__init__: Loading JIT models...")
            self.model.load_jit('{}/llm.text_encoder.j'.format(self.model_dir), '{}/llm.flow_decoder.j'.format(self.model_dir),
                                '{}/llm.final_proj.j'.format(self.model_dir), fp16 if self.fp16 is True else 'fp32')
            print("DEBUG CosyVoice2.__init__: JIT models loaded.")
        if load_trt:
            print("DEBUG CosyVoice2.__init__: Loading TRT models...")
            self.model.load_trt('{}/flow.decoder.estimator.j'.format(self.model_dir), 
                                '{}/flow.decoder.estimator.fp32.onnx'.format(self.model_dir),
                                trt_concurrent, self.fp16)
            print("DEBUG CosyVoice2.__init__: TRT models loaded.")
        
        if configs: 
            del configs
        print("DEBUG CosyVoice2.__init__: __init__ finished.")

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError("inference_instruct is not implemented for CosyVoice2!")

print("DEBUG cosyvoice.py: END OF FILE (after class definitions)")