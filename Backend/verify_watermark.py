# verify_watermark.py
import sys
import json
import os
from pathlib import Path
from audiowmark import WaterMark

# Ensure ffmpeg is accessible, which is crucial for video files.
# Add a common path for system binaries if not already in the PATH.
os.environ["PATH"] += os.pathsep + "/usr/bin"

def extract_watermark_from_file(file_path: str):
    """
    Extracts, decodes, and pretty-prints the secret watermark from a media file.
    """
    print(f"\n--- Attempting to extract watermark from: {Path(file_path).name} ---")
    
    if not os.path.exists(file_path):
        print(f"\n[ FAILED ] Error: File not found at '{file_path}'")
        return

    try:
        # 1. Read the audio data from the file. The library uses ffmpeg
        #    to automatically handle video files and extract the audio stream.
        audio_data = WaterMark.read_audio(file_path)

        # 2. Extract the watermark bytes. We extract a generous amount (256 bytes)
        #    to ensure we get the full JSON payload.
        extracted_bytes = WaterMark.extract_watermark(audio=audio_data, len_wm_bytes=256)

        # 3. Decode the bytes and clean up any trailing null characters.
        decoded_payload = extracted_bytes.decode('utf-8').rstrip('\x00')

        if not decoded_payload:
            print("\n[ FAILED ] Watermark could not be extracted. The data may be missing or corrupt.")
            return

        print("\n[ SUCCESS ] Raw watermark payload found and decoded:")
        print(f"'{decoded_payload}'")

        # 4. Attempt to parse the payload as JSON for pretty-printing.
        try:
            payload_json = json.loads(decoded_payload)
            print("\n--- Parsed Provenance Data ---")
            for key, value in payload_json.items():
                print(f"  - {key}: {value}")
            print("----------------------------")
        except json.JSONDecodeError:
            print("\n[ INFO ] The payload is valid text but not in the expected JSON format.")

    except Exception as e:
        print(f"\n[ FAILED ] An unexpected error occurred during extraction: {e}")
        print("This could mean no watermark was found, the file is corrupt, or ffmpeg is not installed/accessible.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_watermark.py <path_to_your_video_or_audio_file>")
        sys.exit(1)
    
    media_file_path = sys.argv[1]
    extract_watermark_from_file(media_file_path)