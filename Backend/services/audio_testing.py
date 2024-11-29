import torch
import torchaudio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioTester:
    """A/B testing system for audio processing parameters."""
    
    def __init__(self, audio_processor):
        self.audio_processor = audio_processor
        self.test_results_dir = Path('test_results')
        self.test_results_dir.mkdir(exist_ok=True)
        
    def run_ab_test(
        self,
        audio_path: str,
        target_language: str,
        parameter_sets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run A/B test with different parameter sets.
        
        Args:
            audio_path: Path to test audio file
            target_language: Target language code
            parameter_sets: List of parameter dictionaries to test
        """
        results = []
        
        for i, params in enumerate(parameter_sets):
            try:
                # Update processor parameters
                self.audio_processor.LANGUAGE_PARAMS[target_language].update(params)
                
                # Process audio and get diagnostics
                audio, analysis = self.audio_processor.process_audio(
                    audio_path,
                    target_language=target_language,
                    return_diagnostics=True
                )
                
                # Save processed audio
                output_path = self.test_results_dir / f"test_{i}_{target_language}.wav"
                torchaudio.save(output_path, audio, self.audio_processor.SAMPLE_RATE)
                
                # Store results
                results.append({
                    'parameter_set': params,
                    'analysis': analysis,
                    'output_path': str(output_path)
                })
                
            except Exception as e:
                logger.error(f"Test {i} failed: {str(e)}")
                results.append({
                    'parameter_set': params,
                    'error': str(e)
                })
        
        # Save test results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.test_results_dir / f"test_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump({
                'target_language': target_language,
                'results': results
            }, f, indent=2)
        
        return results

    def compare_results(self, results: List[Dict[str, Any]]) -> str:
        """Generate a comparison report of test results."""
        report = ["A/B Test Comparison Report", "=" * 30, ""]
        
        for i, result in enumerate(results):
            report.append(f"\nTest {i + 1}:")
            report.append("-" * 20)
            
            if 'error' in result:
                report.append(f"Error: {result['error']}")
                continue
            
            report.append("\nParameters:")
            for param, value in result['parameter_set'].items():
                report.append(f"- {param}: {value}")
            
            report.append("\nAnalysis Results:")
            for metric, value in result['analysis']['waveform_analysis'].items():
                report.append(f"- {metric}: {value:.3f}")
            
            report.append(f"\nOutput file: {result['output_path']}")
        
        return "\n".join(report)