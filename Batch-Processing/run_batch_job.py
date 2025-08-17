# Batch-Processing/run_batch_job.py

import argparse
from pathlib import Path
import sys
import torch
import torchaudio
import json
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- THIS IS THE ROBUST PATH CORRECTION ---
# It ensures that no matter where this script is run from on the HPC,
# it can always find your custom 'services' modules inside the 'Backend' directory.
try:
    project_root = Path(__file__).resolve().parent.parent
    backend_services_path = project_root / 'Backend'
    if backend_services_path.is_dir():
        sys.path.insert(0, str(backend_services_path))
        logging.info(f"Added to Python path: {backend_services_path}")
        from services.cascaded_backend import CascadedBackend
        from services.audio_processor import AudioProcessor
    else:
        raise ImportError("Could not find Backend/services directory.")
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Could not import backend modules. {e}")
    sys.exit(1)
# --- END OF PATH CORRECTION ---

def process_single_video(video_path_str: str, output_dir: Path, target_lang: str):
    """
    Loads the backend, processes a single video file, and saves the output.
    """
    video_path = Path(video_path_str)
    # Use a unique ID from the filename, which is more robust for job arrays
    job_id = video_path.stem
    
    logging.info(f"--- Starting Job: {job_id} ---")
    logging.info(f"Source Video: {video_path}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info(f"Target Language: {target_lang}")

    try:
        # 1. Initialize the backend and audio processor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        backend = CascadedBackend(device=device)
        backend.initialize()
        audio_processor = AudioProcessor()

        # 2. Process the input video to get an audio tensor
        # NOTE: For this to work, the full video file must be accessible.
        # This assumes your HPC nodes can access the main file storage.
        audio_tensor = audio_processor.process_audio(str(video_path))
        
        # 3. Run the full translation pipeline
        result = backend.translate_speech(
            audio_tensor=audio_tensor,
            source_lang="eng",
            target_lang=target_lang,
            original_video_path=video_path
        )
        
        # 4. Define output paths and save the results
        # The full pipeline should save the final video, but for now we save the audio & text
        final_audio_tensor = result['audio']
        output_audio_path = output_dir / f"{job_id}.wav"
        output_transcript_path = output_dir / f"{job_id}_transcripts.json"

        torchaudio.save(str(output_audio_path), final_audio_tensor.cpu(), 16000)
        with open(output_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result['transcripts'], f, indent=2, ensure_ascii=False)
            
        logging.info(f"[SUCCESS] Job {job_id} completed.")
        logging.info(f"  - Output audio saved to: {output_audio_path}")
        logging.info(f"  - Transcripts saved to: {output_transcript_path}")

    except Exception as e:
        logging.critical(f"[FAILURE] Job {job_id} failed with an error.", exc_info=True)
        # Write a simple error file for easy debugging of failed jobs
        with open(output_dir / f"{job_id}.error", 'w') as f:
            f.write(f"Error processing {video_path}:\n\n{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single batch processing job for the MCF.")
    parser.add_argument("video_path", help="The full path to the source video file.")
    parser.add_argument("output_dir", type=Path, help="The directory to save the output files.")
    parser.add_argument("--lang", default="fra", help="The target language code (e.g., 'fra', 'spa').")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_single_video(args.video_path, args.output_dir, args.lang)