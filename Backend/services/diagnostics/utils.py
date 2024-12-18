"""
Utility functions for audio diagnostics.
"""
import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

def calculate_spectral_slope(band_content: torch.Tensor) -> float:
    """
    Calculate spectral slope in a frequency band.
    
    Args:
        band_content (torch.Tensor): Frequency band content
        
    Returns:
        float: Spectral slope value
    """
    try:
        x = torch.arange(band_content.shape[1], dtype=torch.float32)
        y = torch.mean(band_content, dim=0)
        
        # Calculate linear regression
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sum((x - x_mean) ** 2)
        
        slope = numerator / (denominator + 1e-8)
        return float(slope)
    except Exception as e:
        logger.error(f"Spectral slope calculation failed: {str(e)}")
        return 0.0

def analyze_peak_regularity(peaks: torch.Tensor) -> float:
    """
    Analyze regularity of peaks in a signal.
    
    Args:
        peaks (torch.Tensor): Peak positions
        
    Returns:
        float: Regularity score between 0 and 1
    """
    try:
        if len(peaks) < 2:
            return 0.0
            
        # Calculate intervals between peaks
        intervals = torch.diff(peaks)
        
        # Calculate regularity as inverse of interval variance
        regularity = 1.0 - torch.std(intervals) / (torch.mean(intervals) + 1e-8)
        return float(min(1.0, max(0.0, regularity)))
    except Exception as e:
        logger.error(f"Peak regularity analysis failed: {str(e)}")
        return 0.0

def find_significant_peaks(signal: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Find significant peaks in a signal.
    
    Args:
        signal (torch.Tensor): Input signal
        threshold (float): Peak detection threshold
        
    Returns:
        List[int]: Indices of significant peaks
    """
    try:
        peaks = []
        for i in range(1, len(signal)-1):
            if (signal[i] > threshold * torch.max(signal) and
                signal[i] > signal[i-1] and 
                signal[i] > signal[i+1]):
                peaks.append(i)
        return peaks
    except Exception as e:
        logger.error(f"Peak finding failed: {str(e)}")
        return []

def calculate_frequency_bands(spec: torch.Tensor, sample_rate: int = 16000) -> Dict[str, float]:
    """
    Calculate energy in different frequency bands.
    
    Args:
        spec (torch.Tensor): Spectrogram
        sample_rate (int): Audio sample rate
        
    Returns:
        Dict[str, float]: Energy in each frequency band
    """
    try:
        n_bins = spec.shape[1]
        freq_per_bin = sample_rate / (2 * n_bins)
        
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mids': (250, 500),
            'mids': (500, 2000),
            'high_mids': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
        
        band_energies = {}
        for band_name, (low_freq, high_freq) in bands.items():
            low_bin = int(low_freq / freq_per_bin)
            high_bin = min(int(high_freq / freq_per_bin), n_bins)
            
            if low_bin < high_bin:
                band_energy = float(torch.mean(spec[:, low_bin:high_bin]).item())
                band_energies[band_name] = band_energy
            else:
                band_energies[band_name] = 0.0
                
        return band_energies
    except Exception as e:
        logger.error(f"Frequency band calculation failed: {str(e)}")
        return {band: 0.0 for band in ['sub_bass', 'bass', 'low_mids', 'mids', 'high_mids', 'presence', 'brilliance']}

def analyze_temporal_stability(signal: torch.Tensor, 
                             window_size: int = 1024) -> float:
    """
    Analyze temporal stability of a signal.
    
    Args:
        signal (torch.Tensor): Input signal
        window_size (int): Analysis window size
        
    Returns:
        float: Stability score between 0 and 1
    """
    try:
        if len(signal) < window_size:
            return 0.0
            
        # Split signal into windows
        n_windows = len(signal) // window_size
        windows = signal[:n_windows * window_size].reshape(n_windows, window_size)
        
        # Calculate RMS for each window
        rms_values = torch.sqrt(torch.mean(windows ** 2, dim=1))
        
        # Calculate stability as inverse of RMS variance
        stability = 1.0 - torch.std(rms_values) / (torch.mean(rms_values) + 1e-8)
        return float(min(1.0, max(0.0, stability)))
    except Exception as e:
        logger.error(f"Temporal stability analysis failed: {str(e)}")
        return 0.0

def extract_pitch_contour(audio: torch.Tensor, 
                         sample_rate: int = 16000, 
                         window_size: int = 1024,
                         hop_length: int = 256) -> np.ndarray:
    """
    Extract pitch contour using autocorrelation.
    
    Args:
        audio (torch.Tensor): Input audio
        sample_rate (int): Audio sample rate
        window_size (int): Analysis window size
        hop_length (int): Hop length between windows
        
    Returns:
        np.ndarray: Pitch contour in Hz
    """
    try:
        audio_np = audio.squeeze().numpy()
        n_frames = (len(audio_np) - window_size) // hop_length + 1
        pitch_contour = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = audio_np[i*hop_length:i*hop_length+window_size]
            
            # Apply window function
            frame = frame * np.hanning(len(frame))
            
            # Compute autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find pitch period
            peaks = find_autocorr_peaks(corr)
            if peaks and corr[peaks[0]] > 0.1:
                pitch_contour[i] = sample_rate / peaks[0]
                
        return pitch_contour
    except Exception as e:
        logger.error(f"Pitch contour extraction failed: {str(e)}")
        return np.array([])

def find_autocorr_peaks(corr: np.ndarray, 
                       min_period: int = 20, 
                       max_period: int = 1000) -> List[int]:
    """
    Find peaks in autocorrelation function for pitch detection.
    
    Args:
        corr (np.ndarray): Autocorrelation function
        min_period (int): Minimum period in samples
        max_period (int): Maximum period in samples
        
    Returns:
        List[int]: Peak positions
    """
    try:
        peaks = []
        for i in range(min_period, min(max_period, len(corr)-1)):
            if (corr[i] > corr[i-1] and corr[i] > corr[i+1] and
                corr[i] > 0.5 * np.max(corr[min_period:max_period])):
                peaks.append(i)
                if len(peaks) == 1:  # Only need first strong peak for pitch
                    break
        return peaks
    except Exception as e:
        logger.error(f"Autocorrelation peak finding failed: {str(e)}")
        return []

def calculate_zero_crossing_rate(audio: torch.Tensor) -> float:
    """
    Calculate zero crossing rate of audio signal.
    
    Args:
        audio (torch.Tensor): Input audio signal
        
    Returns:
        float: Zero crossing rate
    """
    try:
        audio_np = audio.squeeze().numpy()
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_np))))
        rate = zero_crossings / len(audio_np)
        return float(rate)
    except Exception as e:
        logger.error(f"Zero crossing rate calculation failed: {str(e)}")
        return 0.0

def calculate_spectral_flatness(spec: torch.Tensor) -> float:
    """
    Calculate spectral flatness (Wiener entropy).
    
    Args:
        spec (torch.Tensor): Magnitude spectrogram
        
    Returns:
        float: Spectral flatness value
    """
    try:
        spec = spec + 1e-8  # Avoid log(0)
        geometric_mean = torch.exp(torch.mean(torch.log(spec)))
        arithmetic_mean = torch.mean(spec)
        flatness = geometric_mean / arithmetic_mean
        return float(flatness)
    except Exception as e:
        logger.error(f"Spectral flatness calculation failed: {str(e)}")
        return 0.0