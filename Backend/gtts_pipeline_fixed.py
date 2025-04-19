# gtts_pipeline_fixed.py

import os
import whisper
import numpy as np
import librosa
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS

class SimpleSpeechTranslationPipeline:
    def __init__(self):
        # Initialize ASR (Whisper)
        print("Loading Whisper model...")
        self.asr_model = whisper.load_model("medium")
        
        # Initialize Translation (NLLB)
        print("Loading NLLB translation model...")
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        
        print("Pipeline initialized successfully!")

    def process(self, audio_path, target_lang="es", output_path=None):
        """Full pipeline from speech to translated speech"""
        if output_path is None:
            output_path = f"output_{target_lang}.mp3"
            
        # Step 1: Speech recognition
        print(f"Transcribing audio: {audio_path}")
        text, source_lang = self.transcribe(audio_path)
        print(f"Transcribed text ({source_lang}): {text}")
        
        # Step 2: Translation
        print(f"Translating from {source_lang} to {target_lang}")
        translated_text = self.translate(text, source_lang, target_lang)
        print(f"Translated text: {translated_text}")
        
        # Step 3: Speech synthesis
        print(f"Generating speech")
        output_file = self.synthesize_speech(translated_text, target_lang, output_path)
        print(f"Output saved to: {output_file}")
        
        return {
            "source_text": text,
            "source_lang": source_lang,
            "translated_text": translated_text,
            "target_lang": target_lang,
            "output_file": output_file
        }

    def transcribe(self, audio_path):
        """Transcribe speech to text using Whisper"""
        try:
            # Load audio file using librosa instead of Whisper's loader
            print(f"Loading audio with librosa: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Ensure the audio is in the correct format for Whisper
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono
            
            # Transcribe using Whisper
            print("Running Whisper transcription...")
            result = self.asr_model.transcribe(audio)
            return result["text"], result.get("language", "en")
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            # Fallback to returning empty result
            return "Failed to transcribe audio", "en"

    def translate(self, text, src_lang, tgt_lang):
        """Translate text using NLLB"""
        # Convert language codes to NLLB format
        src_lang_code = self._convert_to_nllb_code(src_lang)
        tgt_lang_code = self._convert_to_nllb_code(tgt_lang)
        
        # Tokenize and translate
        inputs = self.translator_tokenizer(text, return_tensors="pt")
        translated_tokens = self.translator_model.generate(
            **inputs, 
            forced_bos_token_id=self.translator_tokenizer.lang_code_to_id[tgt_lang_code],
            max_length=200
        )
        translated_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text

    def synthesize_speech(self, text, tgt_lang, output_path):
        """Generate speech in target language using gTTS"""
        # Generate speech with gTTS
        tts = gTTS(text=text, lang=self._convert_to_gtts_code(tgt_lang))
        tts.save(output_path)
        return output_path

    def _convert_to_nllb_code(self, lang_code):
        """Convert ISO language code to NLLB language code"""
        mapping = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
        }
        return mapping.get(lang_code, "eng_Latn")  # Default to English

    def _convert_to_gtts_code(self, lang_code):
        """Convert ISO language code to gTTS language code"""
        mapping = {
            "en": "en",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "zh": "zh-CN",
            "ja": "ja",
            "ko": "ko",
        }
        return mapping.get(lang_code, "en")  # Default to English