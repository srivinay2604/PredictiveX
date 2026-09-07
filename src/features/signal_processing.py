import numpy as np
from scipy import stats

def compute_fft_features(signal: np.ndarray, sample_rate: float = 1000.0) -> dict:
    """
    Computes frequency-domain features via Fast Fourier Transform (FFT) for vibration/acoustic signals.
    """
    if len(signal) == 0:
        return {"spectral_centroid": 0.0, "spectral_energy": 0.0, "dominant_frequency": 0.0}
    
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    
    spectral_energy = np.sum(fft_vals ** 2) / n
    dominant_frequency = fft_freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0.0
    
    sum_fft = np.sum(fft_vals)
    spectral_centroid = np.sum(fft_freqs * fft_vals) / sum_fft if sum_fft > 0 else 0.0
    
    return {
        "spectral_centroid": float(spectral_centroid),
        "spectral_energy": float(spectral_energy),
        "dominant_frequency": float(dominant_frequency)
    }

def compute_time_domain_vibration_features(signal: np.ndarray) -> dict:
    """
    Computes time-domain vibration metrics: RMS, Crest Factor, Kurtosis, Peak-to-Peak.
    """
    if len(signal) == 0:
        return {"rms": 0.0, "crest_factor": 0.0, "kurtosis": 0.0, "peak_to_peak": 0.0}
    
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0.0
    kurtosis_val = float(stats.kurtosis(signal)) if len(signal) > 3 else 0.0
    peak_to_peak = np.ptp(signal)
    
    return {
        "rms": float(rms),
        "crest_factor": float(crest_factor),
        "kurtosis": float(kurtosis_val),
        "peak_to_peak": float(peak_to_peak)
    }
