# test_aws_polly.py
import boto3
import os
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment # For potential MP3 to WAV conversion
import librosa # For resampling if needed

# --- Configuration ---
TEXT_TO_SYNTHESIZE_DE = "Hallo Welt, dies ist ein Test der deutschen Sprachsynthese mit Amazon Polly."
TEXT_TO_SYNTHESIZE_EN = "Hello world, this is a test of English speech synthesis with Amazon Polly."

# German Voice - Vicki is a good neural voice in eu-central-1, Hans is another. Marlene is standard.
# Find voice IDs: https://docs.aws.amazon.com/polly/latest/dg/voicelist.html
POLLY_VOICE_ID_DE = "Vicki" # Neural German voice
POLLY_ENGINE_DE = "neural"
# POLLY_VOICE_ID_DE = "Marlene" # Standard German voice
# POLLY_ENGINE_DE = "standard"

# English Voice - Joanna is a good neural US English voice
POLLY_VOICE_ID_EN = "Joanna"
POLLY_ENGINE_EN = "neural"

OUTPUT_SAMPLE_RATE = 16000 # For OpenVoice compatibility later

# --- End Configuration ---

def synthesize_with_polly(text, voice_id, engine, output_filename_base, region="eu-central-1"):
    """Synthesizes text using AWS Polly and saves as MP3, then converts to 16kHz WAV."""
    try:
        print(f"Attempting to synthesize with Polly: VoiceId='{voice_id}', Engine='{engine}', Region='{region}'")
        polly_client = boto3.client('polly', region_name=region)
        
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice_id,
            Engine=engine
        )

        mp3_path = Path(f"{output_filename_base}.mp3")
        wav_path = Path(f"{output_filename_base}_{OUTPUT_SAMPLE_RATE}hz.wav")

        if 'AudioStream' in response:
            with open(mp3_path, 'wb') as f:
                f.write(response['AudioStream'].read())
            print(f"SUCCESS: AWS Polly MP3 audio saved to: {mp3_path} (Size: {mp3_path.stat().st_size} bytes)")

            # Convert MP3 to WAV at target sample rate
            print(f"Converting {mp3_path} to {wav_path} at {OUTPUT_SAMPLE_RATE}Hz...")
            sound = AudioSegment.from_mp3(str(mp3_path))
            sound = sound.set_frame_rate(OUTPUT_SAMPLE_RATE).set_channels(1)
            sound.export(str(wav_path), format="wav")

            if wav_path.exists() and wav_path.stat().st_size > 100:
                print(f"SUCCESS: Converted to 16kHz WAV: {wav_path} (Size: {wav_path.stat().st_size} bytes)")
                os.remove(mp3_path) # Clean up MP3
                return str(wav_path)
            else:
                print(f"ERROR: WAV conversion failed or output file is too small for {wav_path}")
                return None
        else:
            print(f"ERROR: AWS Polly did not return AudioStream. Response: {response}")
            return None

    except Exception as e:
        print(f"ERROR during AWS Polly synthesis or conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("--- Testing AWS Polly Text-to-Speech ---")

    # Test German
    print("\n--- Synthesizing German ---")
    german_output_file = synthesize_with_polly(
        TEXT_TO_SYNTHESIZE_DE, 
        POLLY_VOICE_ID_DE, 
        POLLY_ENGINE_DE,
        "polly_german_output"
    )
    if german_output_file:
        print(f"German synthesis test successful. Output: {german_output_file}")
    else:
        print("German synthesis test FAILED.")

    # Test English
    print("\n--- Synthesizing English ---")
    english_output_file = synthesize_with_polly(
        TEXT_TO_SYNTHESIZE_EN, 
        POLLY_VOICE_ID_EN, 
        POLLY_ENGINE_EN,
        "polly_english_output",
        region="us-east-1" # Joanna is often in us-east-1
    )
    if english_output_file:
        print(f"English synthesis test successful. Output: {english_output_file}")
    else:
        print("English synthesis test FAILED.")

    print("\n--- AWS Polly Test Complete ---")