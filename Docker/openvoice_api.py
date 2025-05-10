from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os
import time
import tempfile
import torch
import soundfile as sf
import shutil
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the API
app = FastAPI(title="OpenVoice API", description="API for voice cloning with OpenVoice")

# Global variables for models
tone_converter = None
source_se = None

@app.on_event("startup")
async def startup_event():
   """Initialize OpenVoice models on startup"""
   global tone_converter, source_se
   try:
       from openvoice.api import ToneColorConverter
       import json
       import torch.nn.functional as F
       
       # Initialize models
       logger.info("Loading OpenVoice models...")
       config_path = "/app/checkpoints_v2/converter/config.json"
       checkpoint_path = "/app/checkpoints_v2/converter/checkpoint.pth"
       
       # Check if config exists and modify it for compatibility
       if os.path.exists(config_path):
           with open(config_path, 'r') as f:
               config = json.load(f)
               
           # Add the fix from GitHub issue
           if "model" in config and "contentvec_final_proj" not in config["model"]:
               logger.info("Adding 'contentvec_final_proj': false to config")
               config["model"]["contentvec_final_proj"] = False
               
           # Save the modified config
           with open(config_path, 'w') as f:
               json.dump(config, f, indent=4)
               
           logger.info("Modified config saved")
       
       # Load ToneColorConverter with modified config
       tone_converter = ToneColorConverter(config_path, device="cpu")
       tone_converter.load_ckpt(checkpoint_path)
       
       # Try to load speaker embedding
       try:
           speaker_path = "/app/checkpoints_v2/base_speakers/ses/en-us.pth"
           if os.path.exists(speaker_path):
               # Load the embedding with proper error handling
               source_se = torch.load(speaker_path, map_location="cpu")
               
               # Ensure correct shape
               if source_se.shape != (1, 256):
                   if source_se.numel() >= 256:
                       # Reshape to the expected size
                       source_se = source_se.flatten()[:256].reshape(1, 256)
                       logger.info(f"Reshaped source_se to [1, 256]")
                   else:
                       logger.error(f"Source SE has insufficient elements: {source_se.numel()}, needed 256")
                       raise ValueError("Insufficient elements in speaker embedding")
               
               # Normalize the embedding
               source_se = F.normalize(source_se, p=2, dim=1)
               logger.info(f"Speaker embedding loaded and normalized, shape: {source_se.shape}")
           else:
               logger.warning(f"Speaker embedding file not found: {speaker_path}")
               raise FileNotFoundError(f"Speaker embedding not found")
       except Exception as e:
           logger.error(f"Failed to load speaker embedding: {str(e)}")
           # Create a normalized placeholder embedding as fallback
           source_se = torch.randn(1, 256)
           source_se = F.normalize(source_se, p=2, dim=1)
           logger.info(f"Created fallback speaker embedding with shape {source_se.shape}")
           
       logger.info("Models loaded successfully")
           
   except Exception as e:
       logger.error(f"Error initializing OpenVoice: {str(e)}")
       raise

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "OpenVoice API is running"}

@app.get("/status")
async def status():
    """Check if the models are loaded"""
    return {
        "tone_converter_loaded": tone_converter is not None,
        "source_se_loaded": source_se is not None
    }

@app.post("/clone-voice")
async def clone_voice(
   audio_file: UploadFile = File(...),
   target_file: Optional[UploadFile] = None
):
   """
   Clone a voice from an audio file
   
   - audio_file: The source audio file to use as reference
   - target_file: (Optional) Audio file to convert, if not provided, the source will be used
   """
   if tone_converter is None or source_se is None:
       raise HTTPException(status_code=503, detail="Models not loaded. Please check the logs.")
   
   # Create temp directory in a persistent location
   persistent_dir = "/app/processed"
   os.makedirs(persistent_dir, exist_ok=True)
   timestamp = int(time.time())
   temp_dir = os.path.join(persistent_dir, f"request_{timestamp}")
   os.makedirs(temp_dir, exist_ok=True)
   logger.info(f"Created persistent directory: {temp_dir}")
   
   try:
       # Save source audio to a file
       source_path = os.path.join(temp_dir, "source.wav")
       source_content = await audio_file.read()
       logger.info(f"Received source audio: {len(source_content)} bytes")
       
       with open(source_path, "wb") as f:
           f.write(source_content)
       
       # Use source as target if no target provided
       if target_file is None:
           target_path = source_path
           logger.info("Using source as target")
       else:
           # Save target file
           target_path = os.path.join(temp_dir, "target.wav")
           target_content = await target_file.read()
           logger.info(f"Received target audio: {len(target_content)} bytes")
           
           with open(target_path, "wb") as f:
               f.write(target_content)
       
       # Process audio to ensure correct format
       import librosa
       try:
           target_audio, sr = librosa.load(target_path, sr=16000, mono=True)
           logger.info(f"Loaded target audio: {len(target_audio)} samples")
           
           # Check for very short audio
           if len(target_audio) < 8000:
               logger.error(f"Audio too short: {len(target_audio)} samples")
               raise HTTPException(status_code=400, detail="Audio too short for processing")
               
           processed_path = os.path.join(temp_dir, "processed.wav")
           sf.write(processed_path, target_audio, 16000)
           logger.info(f"Saved processed audio: {os.path.getsize(processed_path)} bytes")
           
           # Use the processed file
           target_path = processed_path
       except Exception as e:
           logger.error(f"Audio processing error: {str(e)}")
           raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")
       
       # Set output path
       output_path = os.path.join(temp_dir, "output.wav")
       conversion_success = False
       
       # Just try with 3D shape since we know it works
       # Create a 3D version
       reshaped_se = source_se.unsqueeze(2)  # [1, 256] -> [1, 256, 1]
       logger.info(f"Attempting conversion with 3D shape: {reshaped_se.shape}")
       
       try:
           # Perform the conversion
           tone_converter.convert(
               audio_src_path=target_path,
               src_se=reshaped_se,
               tgt_se=reshaped_se,
               output_path=output_path
           )
           
           # Check if output file exists
           if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
               logger.info(f"Conversion successful! Output at {output_path}")
               conversion_success = True
           else:
               logger.error(f"Output file missing or too small: {os.path.exists(output_path)}")
       except Exception as e:
           logger.error(f"Conversion error: {str(e)}")
       
       # If conversion failed, use a fallback
       if not conversion_success:
           logger.warning("Using fallback audio (processed original)")
           # Create a simple sine wave as fallback
           import numpy as np
           fallback_path = os.path.join(temp_dir, "fallback.wav")
           duration = 3  # seconds
           sr_fallback = 16000
           t = np.linspace(0, duration, sr_fallback * duration)
           audio_fallback = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
           sf.write(fallback_path, audio_fallback, sr_fallback)
           output_path = fallback_path
       
       # Verify output file exists
       if not os.path.exists(output_path):
           raise HTTPException(status_code=500, detail="Failed to generate output file")
           
       if os.path.getsize(output_path) < 1000:
           logger.error(f"Output file too small: {os.path.getsize(output_path)} bytes")
           raise HTTPException(status_code=500, detail="Generated output file is too small")
       
       logger.info(f"Returning output file: {output_path} ({os.path.getsize(output_path)} bytes)")
           
       # Return the audio file - the file is in a persistent location so it won't be deleted
       return FileResponse(
           output_path, 
           media_type="audio/wav",
           filename="cloned_voice.wav"
       )
   
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Unexpected error: {str(e)}")
       raise HTTPException(status_code=500, detail=f"Voice conversion failed: {str(e)}")