# translate_audio.py

import argparse
from gtts_openvoice_pipeline import SpeechTranslationPipeline

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Speech-to-Speech Translation with Voice Cloning")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("--target", "-t", default="es", help="Target language code (default: es)")
    parser.add_argument("--output", "-o", help="Output audio file path")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpeechTranslationPipeline()
    
    # Process the audio
    output_file = args.output or f"output_{args.target}.wav"
    result = pipeline.process(args.input, args.target, output_file)
    
    # Display results
    print("\n=== Translation Results ===")
    print(f"Source language: {result['source_lang']}")
    print(f"Source text: {result['source_text']}")
    print(f"Target language: {result['target_lang']}")
    print(f"Translated text: {result['translated_text']}")
    print(f"Output file: {result['output_file']}")

if __name__ == "__main__":
    main()