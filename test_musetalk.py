import requests
import os

# --- Configuration ---
# Make sure these file paths are correct relative to where you run the script.
VIDEO_PATH = "input_video_short.mov" 
AUDIO_PATH = "input_audio_short.wav"
OUTPUT_PATH = "lipsynced_result.mp4"
MUSETALK_API_URL = "http://localhost:8003/lipsync-video/"

# --- Check if input files exist ---
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video file not found at '{VIDEO_PATH}'")
    exit()
if not os.path.exists(AUDIO_PATH):
    print(f"ERROR: Audio file not found at '{AUDIO_PATH}'")
    exit()

print(f"Sending '{VIDEO_PATH}' and '{AUDIO_PATH}' to MuseTalk API...")

try:
    # --- Prepare the request ---
    with open(VIDEO_PATH, "rb") as video_file, open(AUDIO_PATH, "rb") as audio_file:
        files = {
            'video_file': (os.path.basename(VIDEO_PATH), video_file, 'video/mp4'),
            'audio_file': (os.path.basename(AUDIO_PATH), audio_file, 'audio/wav')
        }
        
        # --- Make the API call ---
        response = requests.post(MUSETALK_API_URL, files=files)

    # --- Handle the response ---
    if response.status_code == 200:
        print("SUCCESS! Received response from API.")
        # Save the returned video file
        with open(OUTPUT_PATH, "wb") as f:
            f.write(response.content)
        print(f"Result saved to '{OUTPUT_PATH}'")
    else:
        print(f"ERROR: API returned status code {response.status_code}")
        # Try to print the error detail from the JSON response
        try:
            error_details = response.json()
            print("Error details:", error_details)
        except:
            print("Could not parse error response. Raw response text:")
            print(response.text)

except requests.exceptions.ConnectionError as e:
    print(f"CONNECTION ERROR: Could not connect to the MuseTalk API at {MUSETALK_API_URL}.")
    print("Please ensure the 'musetalk-service' Docker container is running.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")