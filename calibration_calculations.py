import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class CalibrationCalculations:
    def __init__(self):
        pass

    def check_signal_to_noise_ratio(self, signal_data, noise_data, sample_rate, min_required_snr=10):
        """
        Check if the signal-to-noise ratio meets the minimum requirement (10 dB)
        
        Parameters:
        -----------
        signal_data : array
            Time-domain signal with the sound source active
        noise_data : array
            Time-domain background noise (no sound source)
        sample_rate : float
            Sampling rate in Hz
        min_required_snr : float
            Minimum required SNR in dB (default: 10 dB)
            
        Returns:
        --------
        meets_requirement : bool
            True if the SNR meets the minimum requirement at all frequencies
        frequencies : array
            Frequency array in Hz
        snr : array
            Signal-to-noise ratio at each frequency in dB
        problem_frequencies : array
            Frequencies where the SNR is below the requirement
        """
        # Ensure data is 1D
        signal_data = np.asarray(signal_data).flatten()
        noise_data = np.asarray(noise_data).flatten()
        
        # Use the same length for both signals
        min_length = min(len(signal_data), len(noise_data))
        signal_data = signal_data[:min_length]
        noise_data = noise_data[:min_length]
        
        # Apply windowing to reduce spectral leakage
        window = signal.windows.hann(min_length)
        signal_windowed = signal_data * window
        noise_windowed = noise_data * window
        
        # Calculate power spectra
        # We use Welch's method for better estimates of the power spectra
        f, signal_psd = signal.welch(signal_windowed, fs=sample_rate, nperseg=min(8192, min_length))
        _, noise_psd = signal.welch(noise_windowed, fs=sample_rate, nperseg=min(8192, min_length))
        
        # Calculate SNR in dB
        # Add a small number to avoid division by zero
        snr = 10 * np.log10(signal_psd / (noise_psd + 1e-10))
        
        # Find frequencies where SNR is below the requirement
        problem_indices = np.where(snr < min_required_snr)[0]
        problem_frequencies = f[problem_indices]
        
        # Check if all frequencies meet the requirement
        meets_requirement = len(problem_frequencies) == 0
        
        return meets_requirement, f, snr, problem_frequencies

    def plot_snr_results(self, frequencies, snr, min_required_snr=10, valid_freq_range=None):
        """
        Plot the signal-to-noise ratio vs frequency
        
        Parameters:
        -----------
        frequencies : array
            Frequency array in Hz
        snr : array
            Signal-to-noise ratio at each frequency in dB
        min_required_snr : float
            Minimum required SNR in dB (default: 10 dB)
        valid_freq_range : tuple or None
            (f_min, f_max) tuple specifying the valid frequency range
            
        Returns:
        --------
        fig, ax : tuple
            Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine which range to plot
        if valid_freq_range:
            f_min, f_max = valid_freq_range
            valid_indices = (frequencies >= f_min) & (frequencies <= f_max)
            plot_freqs = frequencies[valid_indices]
            plot_snr = snr[valid_indices]
        else:
            plot_freqs = frequencies
            plot_snr = snr
        
        # Plot SNR
        ax.semilogx(plot_freqs, plot_snr, 'b-', linewidth=2, label='Signal-to-Noise Ratio')
        
        # Add horizontal line at minimum required SNR
        ax.axhline(y=min_required_snr, color='r', linestyle='--', 
                label=f'Minimum Required SNR ({min_required_snr} dB)')
        
        # Fill areas where SNR is below the requirement
        below_req = plot_snr < min_required_snr
        if np.any(below_req):
            ax.fill_between(plot_freqs, plot_snr, min_required_snr, 
                        where=below_req, color='r', alpha=0.3,
                        label='Below Requirement')
        
        ax.grid(True, which="both", ls="--")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('Signal-to-Noise Ratio Analysis')
        
        if valid_freq_range:
            ax.set_xlim(valid_freq_range)
        
        ax.legend()
        fig.tight_layout()
        
        return fig, ax

    def verify_measurement_conditions(self, signal_data, noise_data, sample_rate, tube_diameter, mic_spacing, min_required_snr=10):
        """
        Verify that measurement conditions meet the requirements of ISO 10534-2
        
        Parameters:
        -----------
        signal_data : array
            Time-domain signal with the sound source active
        noise_data : array
            Time-domain background noise (no sound source)
        sample_rate : float
            Sampling rate in Hz
        tube_diameter : float
            Inner diameter of impedance tube in meters
        mic_spacing : float
            Distance between microphones in meters
        min_required_snr : float
            Minimum required SNR in dB (default: 10 dB)
            
        Returns:
        --------
        meets_snr_requirement : bool
            True if the SNR meets the minimum requirement in the valid frequency range
        fig, ax : tuple
            Figure and Axes objects with the SNR plot
        """
        # Calculate valid frequency range
        c = 343.0  # Speed of sound in air (m/s) at 20°C
        f_min = 0.05 * c / mic_spacing
        f_max = min(0.586 * c / tube_diameter, 0.45 * c / mic_spacing)
        valid_freq_range = (f_min, f_max)
        
        print(f"Valid frequency range: {f_min:.1f} Hz - {f_max:.1f} Hz")
        
        # Check SNR
        meets_requirement, frequencies, snr, problem_frequencies = self.check_signal_to_noise_ratio(
            signal_data, noise_data, sample_rate, min_required_snr
        )
        
        # Filter problem frequencies to include only those in the valid range
        valid_problem_freqs = problem_frequencies[
            (problem_frequencies >= f_min) & (problem_frequencies <= f_max)
        ]
        
        # Check if SNR meets requirement in the valid frequency range
        meets_snr_requirement = len(valid_problem_freqs) == 0
        
        # Create SNR plot
        fig, ax = self.plot_snr_results(frequencies, snr, min_required_snr, valid_freq_range)
        
        # Print results
        if meets_snr_requirement:
            print(f"✅ Signal-to-noise ratio meets the requirement (>{min_required_snr} dB) in the valid frequency range.")
        else:
            print(f"❌ Signal-to-noise ratio does not meet the requirement (>{min_required_snr} dB) at some frequencies.")
            print(f"   Problem frequencies in valid range: {len(valid_problem_freqs)}")
            
            # Print a few examples of problem frequencies
            if len(valid_problem_freqs) > 0:
                print("   Examples (Hz): ", end="")
                for i, f in enumerate(valid_problem_freqs[:5]):
                    print(f"{f:.1f}", end=", " if i < min(4, len(valid_problem_freqs)-1) else "")
                if len(valid_problem_freqs) > 5:
                    print(f"... (and {len(valid_problem_freqs)-5} more)")
                else:
                    print()
        
        return meets_snr_requirement, (fig, ax)
    

    def get_calibration_factor_switching_method(self, H12_normal, H12_switched):
        """
        Apply the switching method calibration to correct for microphone mismatch
        
        Parameters:
        -----------
        H12_normal : array
            Complex transfer function with microphones in normal position
        H12_switched : array
            Complex transfer function with microphones in switched position
        
        Returns:
        --------
        H12_corrected : array
            Corrected complex transfer function
        """
        # Calculate the geometric mean of the two measurements
        # We take the square root of the product of the two transfer functions

        Hc = np.sqrt(H12_normal * H12_switched)
        
        return Hc
    
    def apply_calibration_factor(self, H12, Hc):
        """
        Apply the predetermined calibration factor to correct a transfer function
        
        Parameters:
        -----------
        H12 : array
            Measured complex transfer function to be corrected
        Hc : array
            Complex calibration factor
        
        Returns:
        --------
        H12_corrected : array
            Corrected complex transfer function
        """

        H12_corrected = H12 / Hc
        return H12_corrected
    


