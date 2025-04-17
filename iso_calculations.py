import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window

class CalculationISO:
    def __init__(self):
        self

    def process_time_domain_data(self, mic1_signal, mic2_signal, sample_rate, window_type='hann', block_size=10000, overlap_ratio=0.75):
        """
        Process time-domain signals from two microphones using STFT to compute the transfer function (H1 method)
        
        Parameters:
        -----------
        mic1_signal : array
            Time-domain signal from microphone 1 (closer to the sample)
        mic2_signal : array
            Time-domain signal from microphone 2 (farther from the sample)
        sample_rate : float
            Sampling rate in Hz
        window_type : str
            Window type for STFT (default: 'hann')
        block_size : int
            Size of each STFT window (in samples)
        overlap_ratio : float
            Overlap between blocks (default: 0.75 → 75%)
                
        Returns:
        --------
        frequencies : array
            Frequency array in Hz
        H12 : array
            Complex transfer function H12(f) = mic2(f) / mic1(f)
        """
        
        # Ensure signals are flattened and the same length
        mic1_signal = np.asarray(mic1_signal).flatten()
        mic2_signal = np.asarray(mic2_signal).flatten()
        min_len = min(len(mic1_signal), len(mic2_signal))
        mic1_signal = mic1_signal[:min_len]
        mic2_signal = mic2_signal[:min_len]
        
        # Calculate STFT using similar parameters as the working function
        window = np.hanning(block_size)
        noverlap = int(block_size * overlap_ratio)
        
        # Calculate time and frequency parameters for consistency
        dt = 1.0 / sample_rate
        blockTime = block_size * dt
        df = 1.0 / blockTime
        nf = int(np.floor(block_size * 0.5)) + 1
        freq = np.linspace(0, sample_rate * 0.5, nf)
        
        # Perform STFT
        f, t, Z1 = stft(mic1_signal, fs=sample_rate, window=window, nperseg=block_size, 
                        noverlap=noverlap, boundary=None, return_onesided=True)
        _, _, Z2 = stft(mic2_signal, fs=sample_rate, window=window, nperseg=block_size, 
                        noverlap=noverlap, boundary=None, return_onesided=True)
        
        # Calculate averaged spectral components for both signals
        Z1_real_avg = np.mean(Z1.real, axis=1)
        Z1_imag_avg = np.mean(Z1.imag, axis=1)
        Z2_real_avg = np.mean(Z2.real, axis=1)
        Z2_imag_avg = np.mean(Z2.imag, axis=1)
        
        # Calculate H12 using the same method as the working function
        H12_real = np.zeros(nf)
        H12_imag = np.zeros(nf)
        H12_mag = np.zeros(nf)
        H12_phase = np.zeros(nf)
        
        for i in range(0, len(f)):
            x2 = Z1_real_avg[i]
            y2 = Z1_imag_avg[i]
            x1 = Z2_real_avg[i]
            y1 = Z2_imag_avg[i]
            
            # Avoid division by zero
            denominator = (x2*x2 + y2*y2)
            if denominator > 1e-10:  # Small threshold to prevent division by zero
                H12_real[i] = (x1*x2 + y1*y2) / denominator
                H12_imag[i] = (y1*x2 - x1*y2) / denominator
                H12_mag[i] = np.sqrt(H12_real[i]**2 + H12_imag[i]**2)
                H12_phase[i] = np.arctan2(H12_imag[i], H12_real[i])
            else:
                H12_real[i] = 0
                H12_imag[i] = 0
                H12_mag[i] = 0
                H12_phase[i] = 0
        
        # Create complex H12
        H12 = H12_real + 1j * H12_imag
        
        return freq, H12, H12_mag, H12_phase

    def smooth_spectrum(self, spectrum, window_size=5):
        """Apply moving average smoothing to a spectrum"""
        return np.convolve(spectrum, np.ones(window_size)/window_size, mode='same')

    def calculate_valid_frequency_range(self, tube_diameter, mic_spacing, sample_rate):
        """
        Calculate the valid frequency range for measurements
        
        Parameters:
        -----------
        tube_diameter : float
            Inner diameter of the impedance tube in meters
        mic_spacing : float
            Distance between microphone positions in meters
        sample_rate : float
            Sampling rate in Hz
            
        Returns:
        --------
        f_min : float
            Minimum valid frequency in Hz
        f_max : float
            Maximum valid frequency in Hz
        """
        # Speed of sound in air (m/s) at 20°C
        c = 343.0
        
        # Upper frequency limit based on tube diameter (plane wave propagation)
        f_max_tube = 0.586 * c / tube_diameter
        
        # Upper frequency limit based on microphone spacing (spatial aliasing)
        f_max_spacing = 0.45 * c / mic_spacing
        
        # Lower frequency limit based on microphone spacing (phase resolution)
        f_max = 0.05 * c / mic_spacing
        
        # Nyquist frequency
        f_nyquist = sample_rate / 2
        
        # Return the most restrictive limits
        f_min = min(f_max_tube, f_max_spacing, f_nyquist)
        
        return f_max, f_min

    def calculate_absorption_coeficent(self, freq, H, l, s, c=343):
        """
        Calcula o coeficiente de absorção acústica a partir de H (complexo).

        Parâmetros:
        - freq: array de frequências (Hz)
        - H: array de valores complexos H = G12 / G11
        - l: distância da amostra até o centro do microfone mais próximo (em metros)
        - s: distância entre os centros dos microfones (em metros)
        - c: velocidade do som (m/s), padrão: 343 m/s

        Retorna:
        - alpha: array com o coeficiente de absorção
        """
        freq = np.asarray(freq)
        H = np.asarray(H)
        
        Hr = H.real
        Hi = H.imag
        Hr2 = Hr**2
        Hi2 = Hi**2

        k = 2 * np.pi * freq / c
        Hr2 = Hr**2
        Hi2 = Hi**2

        D = 1 + Hr2 + Hi2 - 2 * (Hr * np.cos(k * s) + Hi * np.sin(k * s))

        Rr = (2 * Hr * np.cos(k * (2 * l + s)) - np.cos(2 * k * l) - (Hr2 + Hi2) * np.cos(2 * k * (l + s))) / D
        Ri = (2 * Hr * np.sin(k * (2 * l + s)) - np.sin(2 * k * l) - (Hr2 + Hi2) * np.sin(2 * k * (l + s))) / D

        alpha = 1 - Rr**2 - Ri**2
        
        # Ensure values are within physical limits (sometimes numerical issues can cause values >1)
        alpha = np.clip(alpha, -1, 1)
        
        return alpha

    def plot_results(self, frequencies, alpha, valid_freq_range=None):
        """
        Plot the absorption coefficient vs frequency with valid range highlighted
        
        Parameters:
        -----------
        frequencies : array
            Frequency array in Hz
        alpha : array
            Absorption coefficient array
        valid_freq_range : tuple or None
            (f_min, f_max) tuple specifying valid frequency range
            
        Returns:
        --------
        fig, ax : tuple
            Figure and Axes objects that can be further modified or saved
        """
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(frequencies, alpha, 'r-', alpha=0.3, label='All Data')

        if valid_freq_range:
            f_min, f_max = valid_freq_range
            valid_indices = (frequencies >= f_min) & (frequencies <= f_max)

            print(f"valid idices: {valid_indices}")

            print(f"frequencies with valid indices: {frequencies[valid_indices]} ")

            print(f"alpha at valid indices: {alpha[valid_indices]}")

            ax.plot(frequencies[valid_indices], alpha[valid_indices], 'b-', linewidth=2, 
                    label=f'Valid Range ({int(f_min)}-{int(f_max)} Hz)')
            
            # Add vertical lines at limits
            ax.axvline(x=f_min, color='k', linestyle='--', alpha=0.5)
            ax.axvline(x=f_max, color='k', linestyle='--', alpha=0.5)
                    
        ax.grid(True, which="both", ls="--")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Absorption Coefficient')
        ax.set_title('Sound Absorption Coefficient vs Frequency')
        
        # Set x-axis limits to the valid range if specified
        if valid_freq_range:
            ax.set_xlim(valid_freq_range)
        else:
            ax.set_xlim(min(frequencies), max(frequencies))
            
        ax.set_ylim(-1, 1.1)
        ax.set_xlim(0.1, 2000)
        ax.legend()
        
        fig.tight_layout()
        
        return fig, ax
    
    def plot_calibration_function(self, frequencies, calibration_function):
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(frequencies, calibration_function,label='calibration function')

        ax.grid(True, which="both", ls="--")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('H')
        ax.set_title('H vs Frequency')
            
        ax.set_xlim(0.1, 2000)
        ax.legend()
        
        fig.tight_layout()
        
        return fig, ax


    def process_and_analyze_measurements(self, mic1_signal, mic2_signal, sample_rate, tube_diameter, 
                                        mic_spacing, sample_distance):
        """
        Complete processing workflow from raw signals to absorption coefficient
        
        Parameters:
        -----------
        mic1_signal, mic2_signal : arrays
            Time-domain signals from microphones
        sample_rate : float
            Sampling rate in Hz
        tube_diameter : float
            Inner diameter of impedance tube in meters
        mic_spacing : float
            Distance between microphones in meters
        sample_distance : float
            Distance from sample to nearest microphone in meters
        """
        # Calculate valid frequency range
        f_min, f_max = self.calculate_valid_frequency_range(tube_diameter, mic_spacing, sample_rate)
        print(f"Valid frequency range: {f_min:.1f} Hz - {f_max:.1f} Hz")
        
        # Process time domain data to get transfer function
        frequencies, H12 = self.process_time_domain_data(mic1_signal, mic2_signal, sample_rate)
        
        # Calculate absorption coefficient
        alpha = self.calculate_absorption_coefficient(frequencies, H12, mic_spacing, sample_distance)
        
        return frequencies, alpha, f_min, f_max