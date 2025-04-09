import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class CalculationISO:
    def __init__(self):
        self

    def process_time_domain_data(self, mic1_signal, mic2_signal, sample_rate, window_type='hann'):
        """
        Process time-domain signals from two microphones to get the complex transfer function
        
        Parameters:
        -----------
        mic1_signal : array
            Time-domain signal from microphone 1 (closer to the sample)
        mic2_signal : array
            Time-domain signal from microphone 2 (farther from the sample)
        sample_rate : float
            Sampling rate in Hz
        window_type : str
            Window type for spectral analysis (default: 'hann')
            
        Returns:
        --------
        frequencies : array
            Frequency array in Hz
        H12 : array
            Complex transfer function between microphone positions
        """

        #print("mic1_signal shape:", mic1_signal)
        #print("mic2_signal shape:", mic2_signal)

        mic1_signal = np.asarray(mic1_signal).flatten()
        mic2_signal = np.asarray(mic2_signal).flatten()

        # Ensure signals are the same length
        min_length = min(len(mic1_signal), len(mic2_signal))
        mic1_signal = mic1_signal[:min_length]
        mic2_signal = mic2_signal[:min_length]
        
        # Create window function
        window = signal.get_window(window_type, min_length)
        
        # Apply windowing to reduce spectral leakage
        mic1_windowed = mic1_signal * window
        mic2_windowed = mic2_signal * window
        
        # Calculate FFT
        mic1_fft = np.fft.rfft(mic1_windowed)
        mic2_fft = np.fft.rfft(mic2_windowed)
        
        # Calculate frequencies corresponding to FFT bins
        frequencies = np.fft.rfftfreq(min_length, d=1/sample_rate)
        
        # Calculate the complex transfer function H12 = FFT(mic2) / FFT(mic1)
        # Use cross-spectrum method for better noise immunity
        G12 = np.conj(mic1_fft) * mic2_fft  # Cross spectrum
        G11 = np.conj(mic1_fft) * mic1_fft  # Auto spectrum of mic1
        
        # Apply smoothing (optional)
        G12 = self.smooth_spectrum(G12, window_size=5)
        G11 = self.smooth_spectrum(G11, window_size=5)
        
        # Calculate H12 using the H1 estimator (minimizes noise in reference signal)
        H12 = G12 / G11
        
        # Filter out frequency ranges outside the valid measurement range
        # These ranges are determined by the tube dimensions and microphone spacing
        
        return frequencies, H12

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
        
        return f_min, f_max

    def calculate_absorption_coefficient(self, frequencies, H12, mic_spacing, sample_distance):
        """
        Calculate sound absorption coefficient according to ISO 10534-2
        
        Parameters:
        -----------
        frequencies : array
            Frequency array in Hz
        H12 : array
            Complex transfer function between microphone positions
        mic_spacing : float
            Distance between microphones in meters
        sample_distance : float
            Distance from sample to nearest microphone in meters
            
        Returns:
        --------
        alpha : array
            Absorption coefficient for each frequency
        """
        # Speed of sound in air (m/s) at 20°C
        c = 343.0
        
        # Angular frequency
        omega = 2 * np.pi * frequencies
        
        # Wave number
        k = omega / c
        
        # Calculate the reflection factor
        H_I = np.exp(-1j * k * mic_spacing)
        H_R = np.exp(1j * k * mic_spacing)
        
        # Calculate reflection coefficient
        R = ((H12 - H_I) / (H_R - H12)) * np.exp(2j * k * sample_distance)
        
        # Calculate absorption coefficient
        alpha = 1 - np.abs(R)**2
        
        # Ensure values are within physical limits (sometimes numerical issues can cause values >1)
        alpha = np.clip(alpha, 0, 1)
        
        return alpha

    def plot_results(self, frequencies, alpha, valid_freq_range):
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
                    
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="--")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Absorption Coefficient')
        ax.set_title('Sound Absorption Coefficient vs Frequency')
        
        # Set x-axis limits to the valid range if specified
        if valid_freq_range:
            ax.set_xlim(valid_freq_range)
        else:
            ax.set_xlim(min(frequencies), max(frequencies))
            
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0.1, 10000)
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