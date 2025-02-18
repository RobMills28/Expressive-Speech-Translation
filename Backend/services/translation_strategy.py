from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TranslationStrategy:
    """Selects appropriate translation strategy based on audio characteristics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def select_strategy(self, audio_analysis: Dict) -> Dict:
        """Select translation strategy based on audio characteristics"""
        try:
            # Get basic characteristics
            background_music = audio_analysis.get('background_music', {})
            music_confidence = background_music.get('music_confidence', 0.0)
            has_music = background_music.get('has_background_music', False)
            
            # Simple content type determination
            if has_music or music_confidence > 0.15:
                content_type = "speech_with_music"
                self.logger.info(f"Content Type: speech_with_music (Music confidence: {music_confidence})")
            else:
                content_type = "speech_only"
                self.logger.info(f"Content Type: speech_only (Music confidence: {music_confidence})")
            
            # Return basic strategy
            return {
                'content_type': content_type,
                'heard_characteristics': {
                    'music': {
                        'detected': has_music,
                        'confidence': music_confidence
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {str(e)}")
            return {'content_type': 'speech_only'}  # Safe default