import os
import requests
import tempfile
import soundfile as sf
import numpy as np
import librosa

# Create test audio (a simple sine wave)
def create_test_audio():
    duration = 3  # seconds
    sr = 16000
    t = np.linspace(0, duration, sr * duration)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio, sr

# Create temp directory
temp_dir = tempfile.mkdtemp()
print(f"Created temp directory: {temp_dir}")

# Create and save test audio
print("Creating test audio...")
audio, sr = create_test_audio()
source_path = os.path.join(temp_dir, "test_source.wav")
sf.write(source_path, audio, sr)
print(f"Saved test audio to {source_path}, size: {os.path.getsize(source_path)} bytes")

# Make sure the Docker OpenVoice container is running
print("Testing OpenVoice API status...")
try:
    response = requests.get("http://localhost:8000/status")
    if response.status_code == 200:
        print(f"OpenVoice API is running: {response.json()}")
    else:
        print(f"Error: Status code {response.status_code}")
        print(response.text)
        exit(1)
except Exception as e:
    print(f"Error connecting to OpenVoice API: {str(e)}")
    print("Make sure the Docker container is running.")
    exit(1)

# Send test audio to OpenVoice
print("\nTesting voice cloning...")
try:
    # First, preprocess the audio to ensure proper format
    print("Preprocessing audio...")
    y_source, sr_source = librosa.load(source_path, sr=16000, mono=True)
    source_processed = os.path.join(temp_dir, "source_processed.wav")
    sf.write(source_processed, y_source, 16000)
    print(f"Processed audio saved to {source_processed}, size: {os.path.getsize(source_processed)} bytes")
    
    with open(source_processed, "rb") as f_source:
        files = {
            "audio_file": (os.path.basename(source_processed), f_source, "audio/wav")
        }
        print("Sending request to OpenVoice API...")
        response = requests.post("http://localhost:8000/clone-voice", files=files)
    
    if response.status_code == 200:
        output_path = os.path.join(temp_dir, "cloned_output.wav")
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Voice cloning successful! Output saved to {output_path}")
        print(f"Output file size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {str(e)}")