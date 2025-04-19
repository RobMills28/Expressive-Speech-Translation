# speech_translation_pipeline.py
import os
import torch
import whisper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import soundfile as sf
import tempfile
from pathlib import Path
from openvoice.api import ToneColorConverter
import librosa

class SpeechTranslationPipeline:
    def __init__(self, use_voice_cloning=True):
        # Set up paths
        self.base_dir = "/Users/robmills/audio-translation"
        self.config_path = os.path.join(self.base_dir, "checkpoints_v2", "converter", "config.json")
        self.checkpoint_path = os.path.join(self.base_dir, "checkpoints_v2", "converter", "checkpoint.pth")
        self.speaker_path = os.path.join(self.base_dir, "checkpoints_v2", "base_speakers", "ses", "en-us.pth")
        
        # Use voice cloning or just gTTS
        self.use_voice_cloning = use_voice_cloning
        
        # Initialize ASR (Whisper)
        print("Loading Whisper model...")
        self.asr_model = whisper.load_model("medium")
        
        # Initialize Translation (NLLB)
        print("Loading NLLB translation model...")
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        
        # Initialize OpenVoice (if using voice cloning)
        if self.use_voice_cloning:
            print("Loading OpenVoice model...")
            device = "cpu"
            self.tone_converter = ToneColorConverter(self.config_path, device=device)
            self.tone_converter.load_ckpt(self.checkpoint_path)
            self.source_se = torch.load(self.speaker_path, map_location=device)
            print("OpenVoice loaded successfully")
        
        print("Pipeline initialized successfully!")

    def process(self, audio_path, target_lang="es", output_path=None, reference_audio=None):
        """Full pipeline from speech to translated speech with optional voice cloning"""
        if output_path is None:
            output_path = f"output_{target_lang}.wav"
            
        # Create a temp directory for intermediate files
        temp_dir = Path(tempfile.mkdtemp())
        print(f"Using temp directory: {temp_dir}")
        
        try:
            # Step 1: Speech recognition
            print(f"Transcribing audio: {audio_path}")
            text, source_lang = self.transcribe(audio_path)
            print(f"Transcribed text ({source_lang}): {text}")
            
            # Step 2: Translation
            print(f"Translating from {source_lang} to {target_lang}")
            translated_text = self.translate(text, source_lang, target_lang)
            print(f"Translated text: {translated_text}")
            
            # Step 3: Speech synthesis (with or without voice cloning)
            print(f"Generating speech{' with voice cloning' if self.use_voice_cloning else ''}")
            if self.use_voice_cloning:
                # Generate base speech with gTTS
                base_audio_path = temp_dir / "base_audio.mp3"
                base_wav_path = temp_dir / "base_audio.wav"
                
                # Generate with gTTS first
                tts = gTTS(text=translated_text, lang=self._convert_to_gtts_code(target_lang))
                tts.save(str(base_audio_path))
                
                # Convert to WAV
                y, sr = librosa.load(str(base_audio_path), sr=None)
                sf.write(str(base_wav_path), y, sr)
                
                # Apply voice conversion
                if reference_audio is None:
                    reference_audio = audio_path  # Use input audio as reference
                
                # Use OpenVoice to apply voice conversion
                self.tone_converter.convert(
                    audio_src_path=str(base_wav_path),
                    src_se=self.source_se,
                    tgt_se=self.source_se,  # Using same embedding for demo; ideally extract from reference
                    output_path=output_path
                )
            else:
                # Just use gTTS
                tts = gTTS(text=translated_text, lang=self._convert_to_gtts_code(target_lang))
                tts.save(output_path)
            
            print(f"Output saved to: {output_path}")
            
            return {
                "source_text": text,
                "source_lang": source_lang,
                "translated_text": translated_text,
                "target_lang": target_lang,
                "output_file": output_path
            }
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)

    def transcribe(self, audio_path):
        """Transcribe speech to text using Whisper"""
        try:
            # Handle the file path correctly
            if not os.path.isabs(audio_path):
                # Try multiple locations
                possible_paths = [
                    audio_path,  # Current directory
                    os.path.join(self.base_dir, audio_path),  # Base directory
                    os.path.join(self.base_dir, "English (US)", audio_path)  # English (US) folder
                ]
                
                # Use the first path that exists
                for path in possible_paths:
                    if os.path.exists(path):
                        audio_path = path
                        print(f"Found audio file at: {audio_path}")
                        break
                else:
                    print(f"Could not find audio file. Tried: {possible_paths}")
                    return "Could not find audio file", "en"
            
            # Load audio file using librosa
            print(f"Loading audio from: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Ensure the audio is in the correct format
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono
            
            # Transcribe using Whisper
            result = self.asr_model.transcribe(audio)
            return result["text"], result.get("language", "en")
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return "Failed to transcribe audio", "en"

    def translate(self, text, src_lang, tgt_lang):
        """Translate text using NLLB"""
        # Convert language codes to NLLB format
        src_lang_code = self._convert_to_nllb_code(src_lang)
        tgt_lang_code = self._convert_to_nllb_code(tgt_lang)
        
        # Tokenize and translate
        inputs = self.translator_tokenizer(text, return_tensors="pt")
        
        # Find the right token ID method
        forced_bos_token_id = self.translator_tokenizer.convert_tokens_to_ids(tgt_lang_code)
        
        # Generate translation
        translated_tokens = self.translator_model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id,
            max_length=200
        )
        
        translated_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text

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

def main():
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Speech-to-Speech Translation with Voice Cloning")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("--target", "-t", default="es", help="Target language code (default: es)")
    parser.add_argument("--output", "-o", help="Output audio file path")
    parser.add_argument("--reference", "-r", help="Reference audio for voice (defaults to input)")
    parser.add_argument("--no-cloning", action="store_true", help="Disable voice cloning (use gTTS only)")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpeechTranslationPipeline(use_voice_cloning=not args.no_cloning)
    
    # Process the audio
    output_file = args.output or f"output_{args.target}.wav"
    result = pipeline.process(
        args.input, 
        args.target, 
        output_file, 
        args.reference
    )
    
    # Display results
    print("\n=== Translation Results ===")
    print(f"Source language: {result['source_lang']}")
    print(f"Source text: {result['source_text']}")
    print(f"Target language: {result['target_lang']}")
    print(f"Translated text: {result['translated_text']}")
    print(f"Output file: {result['output_file']}")

if __name__ == "__main__":
    main()