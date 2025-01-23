# services/audio_link_routes.py
import yt_dlp
import requests
import tempfile
import os
import re
from pathlib import Path
from flask import jsonify
import logging
from werkzeug.utils import secure_filename
from pydub import AudioSegment

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = {
    'youtube.com': 'YouTube',
    'youtu.be': 'YouTube',
    'tiktok.com': 'TikTok',
    'vm.tiktok.com': 'TikTok'
}

UNSUPPORTED_PLATFORMS = {
    'spotify.com': 'Spotify',
    'netflix.com': 'Netflix',
    'hulu.com': 'Hulu',
    'amazon.com': 'Amazon',
    'disneyplus.com': 'Disney+',
    'soundcloud.com': 'SoundCloud',
    'vimeo.com': 'Vimeo',
    'twitch.tv': 'Twitch',
    'instagram.com': 'Instagram',
    'facebook.com': 'Facebook'
}

def detect_platform(url):
    """Detect platform from URL and return appropriate message"""
    try:
        domain = re.findall(r'(?:www\.)?([\w-]+\.[\w.-]+)', url)[0]
        
        # Check supported platforms
        for platform_domain, platform_name in SUPPORTED_PLATFORMS.items():
            if platform_domain in domain:
                return {
                    'supported': True,
                    'platform': platform_name
                }
        
        # Check known unsupported platforms
        for platform_domain, platform_name in UNSUPPORTED_PLATFORMS.items():
            if platform_domain in domain:
                return {
                    'supported': False,
                    'platform': platform_name,
                    'message': f"Please use a YouTube or TikTok link instead of {platform_name}"
                }
        
        return {
            'supported': False,
            'platform': 'Unknown',
            'message': "Please use a YouTube or TikTok link"
        }
    except Exception as e:
        logger.error(f"Error detecting platform: {str(e)}")
        return {
            'supported': False,
            'platform': 'Unknown',
            'message': "Please check the URL format and try again"
        }

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

def handle_video_platform_url(url, platform):
    """Handle video platform (YouTube/TikTok) URL download and conversion"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, 'output')
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': temp_path,
                'no_playlist': True,  # Important! Prevents playlist processing
                'extract_audio': True,
                'quiet': False,  # Temporarily set to False for debugging
                'max_filesize': 100000000,  # 100MB limit
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # First check if video exists and get info
                    try:
                        video_info = ydl.extract_info(url, download=False)
                        if not video_info:
                            raise Exception("Unable to access video information")
                            
                        # Check duration before downloading
                        duration = video_info.get('duration', 0)
                        if duration and duration > 120:
                            raise Exception("Please use a video that's 2 minutes or shorter")

                        # If checks pass, download the video
                        ydl.download([url])
                    except yt_dlp.utils.DownloadError as e:
                        logger.error(f"yt-dlp download error: {str(e)}")
                        raise Exception("Please check that the video is available and try again")

                # Check for output file
                final_path = temp_path + '.wav'
                if not os.path.exists(final_path):
                    raise Exception("Failed to download audio")

                # Convert to proper format
                converted_wav = os.path.join(temp_dir, 'converted.wav')
                if not convert_to_wav(final_path, converted_wav):
                    raise Exception("Failed to convert audio")

                with open(converted_wav, 'rb') as f:
                    audio_data = f.read()

                return audio_data, 'audio/wav'

            except Exception as e:
                logger.error(f"{platform} processing error: {str(e)}")
                raise Exception(f"Unable to process this {platform} content. Please try a different video")

    except Exception as e:
        logger.error(f"{platform} processing error: {str(e)}")
        raise

def handle_direct_audio(url):
    """Handle direct audio URL download and conversion"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, 'temp_audio')
            
            response = requests.get(url, stream=True)
            if not response.ok:
                raise Exception("Unable to access this audio content. Please check the URL")
                
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            output_wav = os.path.join(temp_dir, 'output.wav')
            if not convert_to_wav(temp_file, output_wav):
                raise Exception("Unable to process this audio format. Please try a different source")
                
            with open(output_wav, 'rb') as f:
                audio_data = f.read()
                
            return audio_data, 'audio/wav'
            
    except Exception as e:
        logger.error(f"Direct audio processing error: {str(e)}")
        raise

def handle_audio_url_processing(url):
    """Main function to process audio URLs"""
    try:
        logger.info(f"Processing URL: {url}")
        
        if not url or not url.strip():
            return jsonify({'error': 'Please provide a URL'}), 400

        platform_info = detect_platform(url)
        if not platform_info['supported']:
            logger.info(f"Unsupported platform attempt: {platform_info['platform']}")
            return jsonify({
                'error': platform_info['message'],
                'errorType': 'platform_unsupported',
                'platform': platform_info['platform']
            }), 400

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                if any(domain in url for domain in ['youtube.com', 'youtu.be']):
                    logger.info("Processing YouTube URL")
                    audio_data, mime_type = handle_video_platform_url(url, 'youtube')
                elif 'tiktok.com' in url:
                    logger.info("Processing TikTok URL")
                    audio_data, mime_type = handle_video_platform_url(url, 'tiktok')
                else:
                    logger.info("Processing direct audio URL")
                    audio_data, mime_type = handle_direct_audio(url)

                logger.info("Successfully processed audio")
                return {
                    'audio_data': audio_data,
                    'mime_type': mime_type
                }

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Processing error: {error_msg}")
                
                if 'exceeds maximum duration' in error_msg:
                    return jsonify({
                        'error': "Please use a video that's 2 minutes or shorter",
                        'errorType': 'duration_exceeded'
                    }), 400
                
                return jsonify({
                    'error': error_msg,
                    'errorType': 'processing_error'
                }), 400

    except Exception as e:
        logger.error(f"Unexpected error in URL processing: {str(e)}")
        return jsonify({
            'error': 'Something went wrong. Please try again',
            'errorType': 'system_error'
        }), 500