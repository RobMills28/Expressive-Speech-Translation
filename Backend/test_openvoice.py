# Modify test_openvoice.py
import torch
import os
import sys
from openvoice.api import ToneColorConverter

# Get absolute paths
base_dir = "/Users/robmills/audio-translation"
config_path = os.path.join(base_dir, "checkpoints_v2", "converter", "config.json")
checkpoint_path = os.path.join(base_dir, "checkpoints_v2", "converter", "checkpoint.pth")
speaker_path = os.path.join(base_dir, "checkpoints_v2", "base_speakers", "ses", "en-us.pth")

print(f"Using config: {config_path}")
print(f"Using checkpoint: {checkpoint_path}")

# Load model with absolute paths
device = "cpu"
try:
    tone_color_converter = ToneColorConverter(config_path, device=device)
    tone_color_converter.load_ckpt(checkpoint_path)
    print("OpenVoice model loaded successfully!")
    
    # Try loading a speaker embedding
    print(f"Loading speaker embedding: {speaker_path}")
    source_se = torch.load(speaker_path, map_location=device)
    print("Speaker embedding loaded successfully!")
    
    print("\nAll components loaded successfully. OpenVoice is ready to use!")
except Exception as e:
    print(f"Error: {str(e)}")
    print(f"Python path: {sys.path}")