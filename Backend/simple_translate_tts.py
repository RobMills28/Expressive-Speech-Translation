# simple_translate_tts.py

import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS

def translate_text(text, src_lang="en", tgt_lang="es"):
    """Translate text using NLLB"""
    print(f"Loading NLLB translation model...")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    
    # Convert language codes to NLLB format
    src_lang_code = convert_to_nllb_code(src_lang)
    tgt_lang_code = convert_to_nllb_code(tgt_lang)
    
    print(f"Translating from {src_lang} to {tgt_lang}...")
    
    # Tokenize the input text
    inputs = translator_tokenizer(text, return_tensors="pt")
    
    # Generate translation
    translated_tokens = translator_model.generate(
        **inputs, 
        forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids(tgt_lang_code),
        max_length=200
    )
    
    # Decode the generated tokens
    translated_text = translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

def text_to_speech(text, lang, output_file):
    """Generate speech from text using gTTS"""
    print(f"Generating speech in {lang}...")
    tts = gTTS(text=text, lang=convert_to_gtts_code(lang))
    tts.save(output_file)
    print(f"Speech saved to: {output_file}")
    return output_file

def convert_to_nllb_code(lang_code):
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

def convert_to_gtts_code(lang_code):
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
    # Check if we have input text from command line
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        # Default text if none provided
        input_text = "This is a test of our translation and speech synthesis pipeline. We're bypassing the audio input for now."
    
    target_lang = "es"  # Default to Spanish
    output_file = "translated_output.mp3"
    
    print(f"Input text: {input_text}")
    
    # Translate the text
    translated_text = translate_text(input_text, "en", target_lang)
    print(f"Translated text: {translated_text}")
    
    # Generate speech from the translated text
    output_path = text_to_speech(translated_text, target_lang, output_file)
    
    print("\n=== Process Complete ===")
    print(f"Original text: {input_text}")
    print(f"Translated text: {translated_text}")
    print(f"Output audio: {output_path}")

if __name__ == "__main__":
    main()