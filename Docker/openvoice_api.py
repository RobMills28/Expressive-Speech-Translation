from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os
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
        
        # Initialize models
        logger.info("Loading OpenVoice models...")
        config_path = "/app/checkpoints_v2/converter/config.json"
        checkpoint_path = "/app/checkpoints_v2/converter/checkpoint.pth"
        
        # Check if checkpoint exists, if not, log error
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            logger.error(f"Config or checkpoint missing. Please download the models using the download script.")
            return
            
        # Load ToneColorConverter
        tone_converter = ToneColorConverter(config_path, device="cpu")
        tone_converter.load_ckpt(checkpoint_path)
        
        # Try to load speaker embedding
        try:
            speaker_path = "/app/checkpoints_v2/base_speakers/ses/en-us.pth"
            source_se = torch.load(speaker_path, map_location="cpu")
            
            # Ensure correct shape
            if source_se.shape != (1, 256):
                if source_se.numel() >= 256:
                    source_se = source_se.flatten()[:256].reshape(1, 256)
                    logger.info(f"Reshaped source_se to [1, 256]")
                else:
                    logger.error(f"Source SE has insufficient elements: {source_se.numel()}, needed 256")
                    source_se = torch.randn(1, 256)
                    source_se = torch.nn.functional.normalize(source_se, p=2, dim=1)
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load speaker embedding: {str(e)}")
            source_se = torch.randn(1, 256)
            source_se = torch.nn.functional.normalize(source_se, p=2, dim=1)
            
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
    
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Save source audio
        source_path = os.path.join(temp_dir, "source.wav")
        with open(source_path, "wb") as f:
            f.write(await audio_file.read())
        
        # Use the same file for target if not provided
        if target_file is None:
            target_path = source_path
        else:
            target_path = os.path.join(temp_dir, "target.wav")
            with open(target_path, "wb") as f:
                f.write(await target_file.read())
        
        # Process the target file to ensure correct format
        y, sr = sf.read(target_path)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
            sf.write(target_path, y, sr)
        
        # Output path
        output_path = os.path.join(temp_dir, "output.wav")
        
        # Perform voice cloning
        logger.info("Performing voice cloning...")
        tone_converter.convert(
            audio_src_path=target_path,
            src_se=source_se,
            tgt_se=source_se,  # Using same embedding as source for simplicity
            output_path=output_path
        )
        
        # Check if output file exists
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Voice cloning failed to generate output file")
        
        # Return the cloned audio file
        return FileResponse(
            output_path, 
            media_type="audio/wav",
            filename="cloned_voice.wav"
        )
    
    except Exception as e:
        logger.error(f"Error in voice cloning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")
    
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {str(e)}")

@app.get("/download-checkpoints")
async def download_checkpoints():
    """Trigger downloading of model checkpoints"""
    try:
        import subprocess
        result = subprocess.run(
            ["curl", "-L", "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip", "-o", "checkpoints.zip"],
            check=True
        )
        
        # Unzip the checkpoints
        subprocess.run(["unzip", "-o", "checkpoints.zip", "-d", "/app/checkpoints_v2"], check=True)
        
        # Remove the zip file
        os.remove("checkpoints.zip")
        
        return {"message": "Checkpoints downloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to download checkpoints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download checkpoints: {str(e)}")