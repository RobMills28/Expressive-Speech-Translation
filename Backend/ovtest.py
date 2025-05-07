import requests
import os

# Use a file in the same directory as the script
audio_file_path = "harvard.wav"  # Just the filename, no path

# Verify the file exists before trying to open it
if not os.path.exists(audio_file_path):
    print(f"ERROR: The file {audio_file_path} does not exist in this directory.")
    print(f"Current directory: {os.getcwd()}")
    print("Please copy the file to this directory and try again.")
    exit(1)

# Send the request
try:
    with open(audio_file_path, "rb") as f:
        files = {"audio_file": ("harvard.wav", f, "audio/wav")}
        print("Sending request to OpenVoice API...")
        response = requests.post("http://localhost:8000/clone-voice", files=files)

    # Save the result
    if response.status_code == 200:
        with open("cloned_voice.wav", "wb") as f:
            f.write(response.content)
        print("Voice cloning successful! Output saved to cloned_voice.wav")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error during API call: {str(e)}")