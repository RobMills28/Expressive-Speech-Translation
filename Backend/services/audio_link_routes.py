# services/audio_link_routes.py
import yt_dlp
import requests
import tempfile
import os
from pathlib import Path
from flask import jsonify
import logging
from werkzeug.utils import secure_filename
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def convert_to_wav(input_path, output_path):
    """Convert any audio file to WAV format with specific parameters"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format='wav')
        return True
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return False

def handle_youtube_url(url):
    """Handle YouTube URL download and conversion"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_wav = os.path.join(temp_dir, 'output.wav')
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': temp_wav,
                'max_filesize': 100000000,  # 100MB limit
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            if not os.path.exists(temp_wav):
                raise Exception("Failed to download audio")
                
            # Convert to proper format
            converted_wav = os.path.join(temp_dir, 'converted.wav')
            if not convert_to_wav(temp_wav, converted_wav):
                raise Exception("Failed to convert audio")
                
            with open(converted_wav, 'rb') as f:
                audio_data = f.read()
                
            return audio_data, 'audio/wav'
            
    except Exception as e:
        logger.error(f"YouTube processing error: {str(e)}")
        raise

def handle_direct_audio(url):
    """Handle direct audio URL download and conversion"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, 'temp_audio')
            
            # Download file
            response = requests.get(url, stream=True)
            if not response.ok:
                raise Exception("Failed to download audio file")
                
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            # Convert to WAV
            output_wav = os.path.join(temp_dir, 'output.wav')
            if not convert_to_wav(temp_file, output_wav):
                raise Exception("Failed to convert audio")
                
            with open(output_wav, 'rb') as f:
                audio_data = f.read()
                
            return audio_data, 'audio/wav'
            
    except Exception as e:
        logger.error(f"Direct audio processing error: {str(e)}")
        raise

def handle_audio_url_processing(url):
    try:
        if 'youtube.com' in url or 'youtu.be' in url:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, 'output')  # Remove .wav extension here
                
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'outtmpl': output_path,
                }

                # Download the audio
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                # The actual output will be output.wav
                final_path = output_path + '.wav'
                
                # Check if file exists and has content
                if not os.path.exists(final_path):
                    logger.error(f"Output file not found at {final_path}")
                    return jsonify({'error': 'Failed to download audio'}), 500

                # Read the file and return it
                with open(final_path, 'rb') as f:
                    audio_data = f.read()
                
                return {
                    'audio_data': audio_data,
                    'mime_type': 'audio/wav'
                }

        return jsonify({'error': 'Unsupported URL type'}), 400

    except Exception as e:
        logger.error(f"URL processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500