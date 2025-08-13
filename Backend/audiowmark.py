# audiowmark.py
"""
Audio watermarking implementation using ffmpeg metadata for provenance tracking.
"""

import subprocess
import tempfile
import os
import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

class WaterMark:
    """Audio watermarking using metadata embedding"""
    
    @staticmethod
    def read_audio(file_path: Union[str, Path]) -> str:
        """Return the file path for subsequent operations"""
        file_path = str(Path(file_path).resolve())
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        return file_path
    
    @staticmethod
    def add_watermark(audio_path: str, payload_bytes: bytes) -> str:
        """Add watermark as metadata to audio file"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Decode payload to string
            payload_str = payload_bytes.decode('utf-8')
            
            # Use ffmpeg to copy audio and add metadata
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-c', 'copy',
                '-metadata', f'comment={payload_str}',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
                
            if not os.path.exists(output_path):
                raise RuntimeError("Watermarking failed - no output file created")
                
            logger.info(f"Watermark embedded as metadata")
            return output_path
            
        except Exception as e:
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise RuntimeError(f"Watermarking failed: {e}")
    
    @staticmethod
    def write_audio(watermarked_path: str, destination: Union[str, Path]) -> None:
        """Move watermarked audio to final destination"""
        destination = str(Path(destination))
        
        if watermarked_path != destination:
            import shutil
            shutil.move(watermarked_path, destination)
        
        logger.info(f"Watermarked audio written to: {destination}")
    
    @staticmethod
    def extract_watermark(audio_path: str, len_wm_bytes: int = 256) -> bytes:
        """Extract watermark from audio metadata"""
        try:
            # Use ffprobe to extract metadata
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            # Parse JSON output
            metadata = json.loads(result.stdout)
            comment = metadata.get('format', {}).get('tags', {}).get('comment', '')
            
            if comment:
                watermark_bytes = comment.encode('utf-8')
                # Pad or truncate to requested length
                if len(watermark_bytes) < len_wm_bytes:
                    watermark_bytes += b'\x00' * (len_wm_bytes - len(watermark_bytes))
                else:
                    watermark_bytes = watermark_bytes[:len_wm_bytes]
                return watermark_bytes
            
            # No watermark found
            logger.warning("No watermark found in audio metadata")
            return b'\x00' * len_wm_bytes
            
        except Exception as e:
            logger.warning(f"Watermark extraction failed: {e}")
            return b'\x00' * len_wm_bytes