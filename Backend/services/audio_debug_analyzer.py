# services/audio_debug_analyzer.py - NEW FILE
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AudioDebugAnalyzer:
    """
    Utility class for debugging audio placement and temporal mapping issues
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def analyze_audio_placement(self, audio: np.ndarray, 
                              process_id: str,
                              debug_dir: Path = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of audio placement and content
        """
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
        
        duration = len(audio) / self.sample_rate
        
        # Basic statistics
        rms_energy = np.sqrt(np.mean(audio ** 2))
        peak_amplitude = np.max(np.abs(audio))
        
        # Silence detection
        silence_threshold = 0.001
        non_silent_samples = np.where(np.abs(audio) > silence_threshold)[0]
        
        analysis = {
            'total_duration': duration,
            'total_samples': len(audio),
            'rms_energy': rms_energy,
            'peak_amplitude': peak_amplitude,
            'silence_threshold': silence_threshold
        }
        
        if len(non_silent_samples) > 0:
            content_start = non_silent_samples[0] / self.sample_rate
            content_end = non_silent_samples[-1] / self.sample_rate
            content_duration = content_end - content_start
            silence_start = content_start
            silence_end = duration - content_end
            
            analysis.update({
                'has_content': True,
                'content_start_time': content_start,
                'content_end_time': content_end,
                'content_duration': content_duration,
                'leading_silence': silence_start,
                'trailing_silence': silence_end,
                'content_ratio': content_duration / duration if duration > 0 else 0
            })
        else:
            analysis.update({
                'has_content': False,
                'content_start_time': None,
                'content_end_time': None,
                'content_duration': 0,
                'leading_silence': duration,
                'trailing_silence': 0,
                'content_ratio': 0
            })
        
        # Chunk-by-chunk analysis
        chunk_analysis = self._analyze_by_chunks(audio, chunk_duration=1.0)
        analysis['chunk_analysis'] = chunk_analysis
        
        # Log summary
        logger.info(f"[{process_id}] Audio Analysis Summary:")
        logger.info(f"  Duration: {duration:.2f}s ({len(audio)} samples)")
        logger.info(f"  RMS Energy: {rms_energy:.4f}, Peak: {peak_amplitude:.4f}")
        
        if analysis['has_content']:
            logger.info(f"  Content: {content_start:.2f}s - {content_end:.2f}s "
                       f"({content_duration:.2f}s, {analysis['content_ratio']:.1%})")
            logger.info(f"  Silence: {silence_start:.2f}s leading, {silence_end:.2f}s trailing")
        else:
            logger.warning(f"  NO AUDIO CONTENT DETECTED (all below {silence_threshold} threshold)")
        
        # Generate debug visualization if debug directory provided
        if debug_dir and analysis['has_content']:
            self._create_debug_visualization(audio, analysis, process_id, debug_dir)
        
        return analysis
    
    def _analyze_by_chunks(self, audio: np.ndarray, chunk_duration: float = 1.0) -> List[Dict[str, Any]]:
        """
        Analyze audio in chunks to understand distribution
        """
        chunk_samples = int(chunk_duration * self.sample_rate)
        num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples  # Round up
        
        chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(audio))
            chunk = audio[start_idx:end_idx]
            
            start_time = start_idx / self.sample_rate
            end_time = end_idx / self.sample_rate
            
            chunk_rms = np.sqrt(np.mean(chunk ** 2))
            chunk_peak = np.max(np.abs(chunk))
            
            # Determine if chunk has significant content
            has_content = chunk_rms > 0.01 or chunk_peak > 0.05
            
            chunk_info = {
                'chunk_index': i,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'rms_energy': chunk_rms,
                'peak_amplitude': chunk_peak,
                'has_content': has_content,
                'samples': len(chunk)
            }
            
            chunks.append(chunk_info)
        
        # Log chunk summary
        content_chunks = [c for c in chunks if c['has_content']]
        logger.info(f"  Chunk Analysis: {len(content_chunks)}/{len(chunks)} chunks have content")
        
        if content_chunks:
            first_content = content_chunks[0]['start_time']
            last_content = content_chunks[-1]['end_time']
            logger.info(f"  Content spans: {first_content:.2f}s to {last_content:.2f}s")
        
        return chunks
    
    def _create_debug_visualization(self, audio: np.ndarray, 
                                  analysis: Dict[str, Any],
                                  process_id: str,
                                  debug_dir: Path):
        """
        Create a visualization of the audio for debugging
        """
        try:
            plt.figure(figsize=(15, 8))
            
            # Time axis
            time_axis = np.linspace(0, len(audio) / self.sample_rate, len(audio))
            
            # Plot 1: Waveform
            plt.subplot(3, 1, 1)
            plt.plot(time_axis, audio, linewidth=0.5, alpha=0.7)
            plt.title(f'Audio Waveform - {process_id}')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # Highlight content region
            if analysis['has_content']:
                plt.axvspan(analysis['content_start_time'], analysis['content_end_time'], 
                           alpha=0.2, color='green', label='Content Region')
                plt.legend()
            
            # Plot 2: Energy envelope
            plt.subplot(3, 1, 2)
            
            # Calculate energy envelope
            hop_length = 512
            frame_length = 1024
            
            # Ensure audio is long enough for framing
            if len(audio) >= frame_length:
                frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
                energy = np.mean(frames ** 2, axis=0)
                energy_times = librosa.frames_to_time(np.arange(len(energy)), 
                                                    sr=self.sample_rate, hop_length=hop_length)
                
                plt.plot(energy_times, energy, linewidth=1.5, color='red')
                plt.axhline(y=0.001, color='orange', linestyle='--', alpha=0.7, label='Silence Threshold')
                plt.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Content Threshold')
            else:
                # For very short audio, just plot the raw energy
                energy = audio ** 2
                plt.plot(time_axis, energy, linewidth=1.5, color='red')
            
            plt.title('Energy Envelope')
            plt.ylabel('Energy')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 3: Chunk analysis
            plt.subplot(3, 1, 3)
            
            chunk_centers = []
            chunk_energies = []
            chunk_colors = []
            
            for chunk in analysis['chunk_analysis']:
                chunk_centers.append((chunk['start_time'] + chunk['end_time']) / 2)
                chunk_energies.append(chunk['rms_energy'])
                chunk_colors.append('green' if chunk['has_content'] else 'red')
            
            plt.scatter(chunk_centers, chunk_energies, c=chunk_colors, alpha=0.7, s=50)
            plt.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Content Threshold')
            
            plt.title('Chunk Content Analysis')
            plt.xlabel('Time (seconds)')
            plt.ylabel('RMS Energy')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend(['Content Threshold', 'Has Content', 'Silent'])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = debug_dir / f'audio_analysis_{process_id}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[{process_id}] Debug visualization saved: {plot_path.name}")
            
        except Exception as e:
            logger.error(f"[{process_id}] Failed to create debug visualization: {e}")
            plt.close()  # Ensure we don't leave plots open
    
    def compare_before_after(self, original_audio: np.ndarray,
                           mapped_audio: np.ndarray,
                           process_id: str,
                           debug_dir: Path = None) -> Dict[str, Any]:
        """
        Compare audio before and after temporal mapping
        """
        logger.info(f"[{process_id}] Comparing original vs mapped audio")
        
        original_analysis = self.analyze_audio_placement(original_audio, f"{process_id}_original", debug_dir)
        mapped_analysis = self.analyze_audio_placement(mapped_audio, f"{process_id}_mapped", debug_dir)
        
        comparison = {
            'original': original_analysis,
            'mapped': mapped_analysis,
            'duration_change': mapped_analysis['total_duration'] - original_analysis['total_duration'],
            'content_duration_change': (mapped_analysis['content_duration'] - 
                                      original_analysis['content_duration']),
            'energy_ratio': (mapped_analysis['rms_energy'] / 
                           max(original_analysis['rms_energy'], 1e-10))
        }
        
        logger.info(f"[{process_id}] Comparison Results:")
        logger.info(f"  Duration change: {comparison['duration_change']:+.2f}s")
        logger.info(f"  Content duration change: {comparison['content_duration_change']:+.2f}s")
        logger.info(f"  Energy ratio: {comparison['energy_ratio']:.2f}")
        
        return comparison