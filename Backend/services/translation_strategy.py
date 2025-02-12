from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TranslationStrategy:
    """Selects appropriate translation strategy based on audio characteristics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # In translation_strategy.py
    def select_strategy(self, audio_analysis: Dict) -> Dict:
        """Select translation strategy based on audio characteristics"""
        try:
            # Detailed audio analysis logging
            self.logger.info("\n=== DETAILED AUDIO ANALYSIS ===")
        
            # Get music characteristics
            music_confidence = audio_analysis.get('music_confidence', 0.0)
            has_music = audio_analysis.get('has_background_music', False)
            feature_scores = audio_analysis.get('feature_scores', {})
        
            # Log music detection details
            self.logger.info(f"""
    Music Detection:
    - Has Background Music: {has_music}
    - Music Confidence: {music_confidence:.3f}
    - Feature Scores:
    • Spectral Flatness: {feature_scores.get('spectral_flatness', 0.0):.3f}
    • Rhythm Regularity: {feature_scores.get('rhythm_regularity', 0.0):.3f}
    • Bass Presence: {feature_scores.get('bass_presence', 0.0):.3f}
    • Temporal Stability: {feature_scores.get('temporal_stability', 0.0):.3f}
            """)

            # Get audio quality metrics
            metrics = audio_analysis.get('metrics', {})
            self.logger.info(f"""
    Audio Quality Metrics:
    - Audio Clarity: {metrics.get('audio_clarity', 0.0):.3f}
    - Background Noise: {metrics.get('background_noise', 0.0):.3f}
    - Voice Consistency: {metrics.get('voice_consistency', 0.0):.3f}
            """)

            # Determine content type based on music presence
            if has_music and music_confidence > 0.4:
                content_type = "speech_with_music"
                self.logger.info("Content Type Decision: speech_with_music (High music confidence)")
            else:
                content_type = "speech_only"
                self.logger.info("Content Type Decision: speech_only (Low music confidence)")
            
            # Log the decision rationale
            self.logger.info(f"""
    Decision Factors:
    - Music Confidence Threshold: 0.4
    - Current Music Confidence: {music_confidence:.3f}
    - Has Background Music Flag: {has_music}
    - Selected Type: {content_type}
            """)
            
            strategy = {
                'content_type': content_type,
                'heard_characteristics': {
                    'music': {
                        'detected': has_music,
                        'confidence': music_confidence,
                        'feature_scores': feature_scores
                    },
                    'speech': {
                        'consistency': metrics.get('voice_consistency', 1.0),
                        'clarity': metrics.get('audio_clarity', 0.0),
                        'background_noise': metrics.get('background_noise', 0.0)
                    }
                }
            }
        
            return strategy
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {str(e)}")
            return {'content_type': 'speech_only'}  # Safe default