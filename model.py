import time
import sys
from win32com.client import Dispatch
from DataReader import DWDataReader
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from typing import Type
from iso_calculations import CalculationISO
from calibration_calculations import CalibrationCalculations
from scipy.signal import welch

import json

class TubeSetupModel:
    def __init__(self):
        self.data = {
            'mic_spacing': '',
            'mic1_sample': '',
            'mic2_sample': '',
            'tube_diameter': ''
        }
        self.filename = ''

    def save_data(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f)

    def load_data(self, filename):
        try:
            with open(filename, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found")
            self.data = {}
        return self.data

    def set_data(self, mic_spacing, mic1_sample, mic2_sample, tube_diameter):
        self.data['mic_spacing'] = mic_spacing
        self.data['mic1_sample'] = mic1_sample
        self.data['mic2_sample'] = mic2_sample
        self.data['tube_diameter'] = tube_diameter

class Dewesoft:
    def __init__(self):
        self.dw = Dispatch("Dewesoft.App")
        sys.stdout.flush()
        self.dw.Init()
        self.dw.Enabled = 1
        self.dw.Visible = 1

    def __del__(self):
        self.close()

    def set_sample_rate(self, sample_rate: int) -> None:
        self.dw.MeasureSampleRate = sample_rate

    def load_setup(self, setup_path: str) -> None:
        self.dw.LoadSetup(setup_path)

    def set_dimensions(self, width: int, height: int) -> None:
        self.dw.Top = 0
        self.dw.Left = 0
        self.dw.Width = width
        self.dw.Height = height

    def measure(self, seconds: int, filename: str) -> None:
        print(f"Running measurements for {seconds} seconds")
        self.dw.Start()
        full_name = filename + ".dxd"
        self.dw.StartStoring(full_name)
        time.sleep(seconds)
        self.dw.Stop()

    def close(self) -> None:
        print("Closing Dewesoft... ", end="")
        sys.stdout.flush()
        self.dw = None
        print("done.")
        sys.stdout.flush()

class DataStore:
    """Centralized data store for sharing measurement results between models."""
    def __init__(self):
        self.measurement_results = {}
        self.absorption_coef = {}

    def add_result(self, result, name : str):
        self.measurement_results[name] = result

    def get_result_by_name(self,name : str, metric :str) -> np.array:
        return self.measurement_results[name]

    def get_results(self):
        return self.measurement_results
    
    def add_absorption_coef(self, coef: tuple[float, float], name: str) -> None:
        self.absorption_coef[name] = coef

    def get_absorption_coef(self, name: str) -> tuple[float, float]:
        return self.absorption_coef[name]

class ResultsModel:
    def __init__(self, data_store : DataStore):
        self.data_store = data_store
        self.processed_measurements = []  # List of processed measurements
        self.selected_metrics = set()  # Set of selected evaluation metrics

    def get_processed_measurements(self):
        return self.processed_measurements

    def set_processed_measurements(self, measurements):
        self.processed_measurements = measurements

    def get_measurement_results(self):
        return self.data_store.get_results()
    
    def get_measurement_by_name(self, name, metric) -> np.array :
        #print(self.data_store.get_results())
        return self.data_store.get_result_by_name(name, metric)

    def toggle_metric(self, metric_name, selected):
        """Adds or removes a metric based on user selection."""
        if selected:
            self.selected_metrics.add(metric_name)
        else:
            self.selected_metrics.discard(metric_name)

    def get_selected_metrics(self):
        return list(self.selected_metrics)
    
    def generate_plot(self, metric, measument):
        measurement_data = self.get_measurement_by_name(measument, metric)
        #print(measurement_data)

        if metric == "Absorption Coefficient":
            interchanged_measurement = self.get_measurement_by_name("TestAbsorcao_MicTrocado", metric)
            fig = self.absorption_coef_graph(measurement_data=measurement_data,mic_changed=interchanged_measurement)
            return fig
            
        if metric == "Fourier Transform":
            # Ensure signal and time are 1D numpy arrays
            signal_1 = np.asarray(measurement_data['AI A-1']).flatten()
            time = np.asarray(measurement_data['Time']).flatten()

            # Compute sampling interval and length
            dt = 1 / measurement_data['SampleRate']
            N = len(signal_1)

            f, Pxx = welch(signal_1, fs=1/dt, window='hann', nperseg=65536, noverlap=32768)
            mask = (f >= 0) & (f <= 1600)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(f[mask], np.sqrt(Pxx[mask]), label='RMS Amplitude (Pa)', color='green')  # sqrt to get Pa from Pa²
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude (Pa)")
            ax.set_title("Welch FFT (0 to 1000 Hz)")
            ax.grid()
            ax.legend()
            return fig
        
        if metric == "original_data":
            signal_1 = measurement_data['AI A-1']
            signal_2 = measurement_data['AI A-2']
            time = measurement_data['Time']

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(time, signal_1, label="Original Signal", color="blue")
            ax.plot(time, signal_2, label="Signal 2 (AI A-2)", color="orange")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Magnitude')
            ax.set_title('Original Signal')
            ax.grid()

            return fig  # Return the figure 

    
    def absorption_coef_graph(self, measurement_data, mic_changed):
        calc = CalculationISO()
        cali = CalibrationCalculations()

        # Ajustar os dados brutos para o mesmo tamanho
        min_length_ch1 = min(len(measurement_data['AI A-1']), len(mic_changed['AI A-1']))
        min_length_ch2 = min(len(measurement_data['AI A-2']), len(mic_changed['AI A-2']))
        
        # Garantir que os dois canais tenham o mesmo comprimento
        min_length = min(min_length_ch1, min_length_ch2)
        
        # Cortar os dados brutos
        measurement_data_adjusted = {
            'AI A-1': measurement_data['AI A-1'][:min_length],
            'AI A-2': measurement_data['AI A-2'][:min_length],
            'SampleRate': measurement_data['SampleRate']
        }
        
        mic_changed_adjusted = {
            'AI A-1': mic_changed['AI A-1'][:min_length],
            'AI A-2': mic_changed['AI A-2'][:min_length],
            'SampleRate': mic_changed['SampleRate']
        }
        
        freq, H12 = calc.process_time_domain_data(
            measurement_data_adjusted['AI A-1'], 
            measurement_data_adjusted['AI A-2'], 
            measurement_data_adjusted["SampleRate"]
        )

        print(f"Frequencies: {freq}")
        
        _, H21 = calc.process_time_domain_data(
            mic_changed_adjusted['AI A-1'], 
            mic_changed_adjusted['AI A-2'], 
            mic_changed_adjusted["SampleRate"]
        )

        Hc = cali.get_calibration_factor_switching_method(H12_normal=H12, H12_switched=H21)

        print(f"Calibration factor {Hc}")

        H12 = cali.apply_calibration_factor(H12,Hc)

        print(f"H12 after calibration: {H12}")

        alpha = calc.calculate_absorption_coefficient(freq, H12, 0.05, 0.1)

        print(f"alpha: {alpha}")

        f_min, f_max = calc.calculate_valid_frequency_range(0.1, 0.05, 0.1)

        print(f"Fmax and Fmin : {f_max} {f_min}")

        fig, ax = calc.plot_results(freq, alpha, valid_freq_range=(f_min, f_max))
        print(fig)
        return fig
        

class MeasurementModel:
    def __init__(self, data_store : DataStore):
        self.data_store = data_store
        self.samples = ["Amostra_x", "Amostra_y"]  # Available samples

    def get_samples(self):
        return self.samples

    def get_measurement_results(self):
        return self.data_store.get_results()

    def add_measurement_result(self, result, name):
        """Adds a measurement result to the results dict."""
        self.data_store.add_result(result, name)




def run():
    dreader = DWDataReader()
    dreader.open_data_file("TestFundo")
    info = dreader.get_measurement_info()
    print(info.sample_rate)
    data = dreader.get_measurements_as_dataframe()
    print(data)



if __name__ == "__main__":
    run()
