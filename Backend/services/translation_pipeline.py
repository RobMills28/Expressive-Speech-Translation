from typing import Dict
import logging
from .translation_environment import TranslationEnvironment

logger = logging.getLogger(__name__)

class TranslationPipeline:
    """
    Manages the complete audio translation pipeline.
    Designed to be model-agnostic for future flexibility.
    """
    
    def __init__(self, model_type: str = "seamless"):
        self.model_type = model_type
        logger.info(f"Initialized translation pipeline with {model_type} model")
    
    def process_audio(self, audio_data: Dict, audio_analysis: Dict) -> Dict:
        """Process audio through the complete pipeline"""
        try:
            # First analyze all audio characteristics
            audio_characteristics = self._analyze_characteristics(audio_analysis)
            logger.info(f"Audio characteristics analyzed: {audio_characteristics}")
            
            # Create translation environment for current model
            environment = TranslationEnvironment.from_characteristics(
                audio_characteristics, 
                model_type=self.model_type
            )
            
            return {
                'characteristics': audio_characteristics,
                'environment': {
                    'type': environment.environment_type,
                    'model': environment.model_type,
                    'speech_prominence': environment.speech_prominence,
                    'music_confidence': environment.music_confidence,
                    'has_music': environment.has_music
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            return {}
            
    def _analyze_characteristics(self, audio_analysis: Dict) -> Dict:
        """Analyze and combine all audio characteristics."""
        try:
            background_music = audio_analysis.get('background_music', {})
            speech_vs_music = background_music.get('speech_vs_music', {})
            
            return {
                'audio_profile': {
                    'has_music': background_music.get('has_background_music', False),
                    'music_confidence': background_music.get('music_confidence', 0.0),
                    'speech_prominence': speech_vs_music.get('speech_prominence', 0.0),
                },
                'noise_profile': {
                    'background_noise': audio_analysis.get('metrics', {}).get('background_noise', 0.0),
                    'voice_consistency': audio_analysis.get('metrics', {}).get('voice_consistency', 0.0),
                    'audio_clarity': audio_analysis.get('metrics', {}).get('audio_clarity', 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Characteristics analysis failed: {str(e)}")
            return {}