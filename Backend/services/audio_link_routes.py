# services/audio_link_routes.py
import yt_dlp
import requests
import tempfile
import os
import re
import random
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
            temp_file = os.path.join(temp_dir, 'output')  # No extension
            
            # Base options used for both YouTube and TikTok
            base_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': temp_file,
                'quiet': False,
                'no_warnings': True,
                'keepvideo': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192'
                }],
                'concurrent_fragment_downloads': 8,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                },
                'nocheckcertificate': True,
                'ignoreerrors': False,  # Keep this False to get proper error messages
                'geo_bypass': True
            }

            if platform == 'tiktok':
                tiktok_opts = {
                    'extractor_args': {
                        'tiktok': {
                            'api_hostname': 'api22-normal-c-alisg.tiktokv.com',
                            'app_name': 'trill',
                            'app_version': '34.1.2',
                            'manifest_app_version': '2023401020',
                            'app_info': '1234567890123456789/trill///1180',
                            'device_id': ''.join(random.choices('0123456789', k=19))
                        }
                    }
                }
                ydl_opts = {**base_opts, **tiktok_opts}
            else:
                ydl_opts = base_opts

            logger.info(f"Using options for {platform}: {ydl_opts}")

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Get video info first
                    info = ydl.extract_info(url, download=False)
                    duration = info.get('duration', 0)
                    
                    if duration > 120:
                        raise Exception("Please use a video that's 2 minutes or shorter")
                        
                    # Download the video
                    ydl.download([url])
                    logger.info(f"Successfully downloaded {platform} content")

                # Look for both possible filenames
                possible_files = [
                    f"{temp_file}.wav",
                    f"{temp_file}.wav.wav",
                    f"{temp_file}.m4a",
                    f"{temp_file}"
                ]

                input_file = None
                for f in possible_files:
                    if os.path.exists(f):
                        input_file = f
                        break

                if not input_file:
                    raise Exception(f"Could not find downloaded audio file. Checked: {possible_files}")

                logger.info(f"Found audio file: {input_file}")
                
                # Convert to proper format
                converted_wav = os.path.join(temp_dir, 'converted.wav')
                success = convert_to_wav(input_file, converted_wav)
                
                if not success:
                    raise Exception("Failed to convert audio format")
                
                if not os.path.exists(converted_wav):
                    raise Exception("Converted file not found")
                
                logger.info(f"Successfully converted audio to WAV")
                
                with open(converted_wav, 'rb') as f:
                    audio_data = f.read()
                
                if not audio_data:
                    raise Exception("Empty audio data")
                
                return audio_data, 'audio/wav'

            except yt_dlp.utils.DownloadError as e:
                logger.error(f"Download error for {platform}: {str(e)}")
                if platform == 'tiktok':
                    raise Exception(f"Unable to access this TikTok content. Please try another video or check the URL.")
                else:
                    raise Exception(f"Unable to access this {platform} content. Please check the URL and try again.")
            
    except Exception as e:
        logger.error(f"{platform} processing error: {str(e)}", exc_info=True)
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