from transformers import SeamlessM4TProcessor, SeamlessM4TForSpeechToSpeech
import torch
import torchaudio
import os
import numpy as np

# Specify the path to your audio file (use raw string)
audio_path = r"/Users/robmills/Documents/Audio Samples/English (US)/Arthur.mp3"

# Load the processor and model
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4TForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")

# Load your audio file
audio, orig_freq = torchaudio.load(audio_path)
audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

# Convert to mono if stereo
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)

# Convert to numpy array and flatten
audio_numpy = audio.squeeze().numpy()

# Process the audio input
inputs = processor(
    audios=audio_numpy,
    sampling_rate=16000,
    src_lang="eng",  # Source language is English
    return_tensors="pt"
)

# Generate the translated speech
with torch.no_grad():
    output = model.generate(**inputs, tgt_lang="fra")  # Target language is French

print(f"Output type: {type(output)}")
print(f"Output structure: {output}")

# Extract the waveform from the output
waveform = output[0].squeeze().cpu().numpy()

# Save the output audio
output_dir = os.path.dirname(audio_path)
output_path = os.path.join(output_dir, "translated_Arthur.wav")

# Convert the numpy array to a torch tensor and ensure it's 2D
waveform_tensor = torch.tensor(waveform).unsqueeze(0)

torchaudio.save(output_path, waveform_tensor, sample_rate=16000)

print(f"Translation complete. Output saved as {output_path}")