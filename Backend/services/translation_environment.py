from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationEnvironment:
    """
    Defines the complete environment for translation based on audio characteristics.
    Specifically tuned for SeamlessM4T parameters while remaining model-agnostic.
    """
    speech_prominence: float = 0.0
    music_confidence: float = 0.0
    has_music: bool = False
    audio_clarity: float = 0.0
    voice_consistency: float = 0.0
    environment_type: str = "general"
    model_type: str = "seamless"

    def get_translation_parameters(self) -> Dict[str, Any]:
        """Get model-specific parameters based on environment characteristics"""
        try:
            # First determine environment type based on audio characteristics
            self.environment_type = self._determine_environment_type()
            
            if self.model_type == "seamless":
                if self.environment_type == "speech_focused":
                    return {
                        'num_beams': 6,
                        'do_sample': False,
                        'max_new_tokens': 8000,
                        'temperature': 0.2,  # Lower temperature for more focused translation
                        'length_penalty': 2.0,
                        'repetition_penalty': 1.5,
                        'no_repeat_ngram_size': 3
                    }
                elif self.environment_type == "mixed_content":
                    return {
                        'num_beams': 4,
                        'do_sample': True,
                        'max_new_tokens': 8000,
                        'temperature': 0.5,  # Balanced temperature
                        'length_penalty': 1.5,
                        'repetition_penalty': 1.2,
                        'no_repeat_ngram_size': 3
                    }
                else:  # general case
                    return {
                        'num_beams': 3,
                        'do_sample': True,
                        'max_new_tokens': 8000,
                        'temperature': 0.7,
                        'length_penalty': 1.0,
                        'repetition_penalty': 1.2,
                        'no_repeat_ngram_size': 3
                    }
            else:
                logger.warning(f"Unknown model type: {self.model_type}, using defaults")
                return {}

        except Exception as e:
            logger.error(f"Error getting translation parameters: {str(e)}")
            return {}

    def _determine_environment_type(self) -> str:
        """
        Determine environment type based on audio characteristics.
        """
        logger.info(f"Determining environment type with speech prominence: {self.speech_prominence}, "
                   f"music confidence: {self.music_confidence}")
        
        if self.speech_prominence > 2.0 and self.audio_clarity > 0.6:
            logger.info("Detected speech-focused environment")
            return "speech_focused"
        elif self.has_music and self.music_confidence > 0.4:
            logger.info("Detected mixed-content environment")
            return "mixed_content"
        else:
            logger.info("Using general environment")
            return "general"