from typing import Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TranslationEnvironment:
    """
    Defines the complete environment for translation based on audio characteristics.
    Model-agnostic by design to support different translation models.
    """
    speech_prominence: float = 0.0
    music_confidence: float = 0.0
    has_music: bool = False
    audio_clarity: float = 0.0
    voice_consistency: float = 0.0
    environment_type: str = "general"
    model_type: str = "seamless"  # Can be changed for different models

    @classmethod
    def from_characteristics(cls, characteristics: Dict, model_type: str = "seamless") -> 'TranslationEnvironment':
        """Create environment from audio characteristics"""
        try:
            env = cls(
                speech_prominence=characteristics['audio_profile']['speech_prominence'],
                music_confidence=characteristics['audio_profile']['music_confidence'],
                has_music=characteristics['audio_profile']['has_music'],
                audio_clarity=characteristics['noise_profile']['audio_clarity'],
                voice_consistency=characteristics['noise_profile']['voice_consistency'],
                model_type=model_type
            )
            
            # Model-agnostic environment determination
            env.environment_type = env._determine_environment_type()
            logger.info(f"Configured {env.model_type} model in {env.environment_type} environment")
            
            return env
            
        except Exception as e:
            logger.error(f"Environment creation failed: {str(e)}")
            return cls()

    def _determine_environment_type(self) -> str:
        """
        Determine environment type based on audio characteristics.
        This logic can be extended for different models.
        """
        if self.speech_prominence > 2.0:
            logger.info(f"Speech-focused environment (prominence: {self.speech_prominence})")
            return "speech_focused"
        elif self.has_music and self.music_confidence > 0.4:
            logger.info(f"Mixed-content environment (music confidence: {self.music_confidence})")
            return "mixed_content"
        else:
            logger.info("General environment")
            return "general"