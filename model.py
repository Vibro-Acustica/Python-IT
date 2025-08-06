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
from scipy.signal import welch, stft, ShortTimeFFT
import pandas as pd

import json

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
        self.chanel_names_in_dewesoft = []
        self.tube_measurement = {}
        self.test_conditions = {}
        self.measurement_results = {}
        self.post_processed_results = {}
        self.absorption_coef = {}
        self.samples = {}
        self.warnings = {}
        self.state = {
            'current_measurement': None,
            'is_measuring': False,
            'is_processing': False,
            'last_error': None,
            'current_sample': None,
            'calibration_status': None,
            'unsaved_changes': False
        }
        self.observers = []

    def add_observer(self, observer):
        """Add an observer to be notified of state changes."""
        self.observers.append(observer)

    def remove_observer(self, observer):
        """Remove an observer."""
        self.observers.remove(observer)

    def notify_observers(self, property_name, value):
        """Notify all observers of a state change."""
        for observer in self.observers:
            observer.on_state_change(property_name, value)

    def set_state(self, property_name, value):
        """Update state and notify observers."""
        self.state[property_name] = value
        self.notify_observers(property_name, value)

    def get_state(self, property_name):
        """Get current state value."""
        return self.state.get(property_name)

    def add_result(self, result, name : str):
        self.measurement_results[name] = result

    def get_result_by_name(self,name : str, metric :str) -> np.array:
        print(f"Getting result by name {name} and metric {metric}")
        return self.measurement_results[name]

    def get_results(self):
        return self.measurement_results
    
    def add_absorption_coef(self, coef: tuple[float, float], name: str) -> None:
        self.absorption_coef[name] = coef

    def get_absorption_coef(self, name: str) -> tuple[float, float]:
        return self.absorption_coef[name]
    
    def add_post_processed(self, data, name: str):
        self.post_processed_results[name] = data

    def get_post_processed(self, name: str):
        return self.post_processed_results[name]
    
    def get_all_post_processed(self):
        return self.post_processed_results
    
    def add_tube_measurements(self, mic_spacing, diameter, mic_to_source_1, mic_to_source_2):
        self.tube_measurement["mic_spac"] = mic_spacing
        self.tube_measurement["diameter"] = diameter
        self.tube_measurement["mic_source_1"] = mic_to_source_1
        self.tube_measurement["mic_source_2"] = mic_to_source_2

    def get_tube_measurements(self):
        return self.tube_measurement
    
    def add_test_conditions(self, temp, humidty, pressure):
        self.test_conditions["temp"] = temp
        self.test_conditions["humi"] = humidty
        self.test_conditions["pressure"] = pressure

    def get_test_conditions(self):
        return self.test_conditions
    
    def add_sample(self, sample, sample_name):
        self.samples[sample_name] = sample

    def get_sample(self, sample_name):
        return self.samples[sample_name]
    
    def add_warnings(self, snr, auto_spec, calib):
        self.warnings["snr"] = snr
        self.warnings["auto_spec"] = auto_spec
        self.warnings["calib"] = calib

    def get_warnings(self):
        return self.warnings
    
    def set_channel_names(self, names: list[str]):
        self.chanel_names_in_dewesoft = names

    def get_channel_names(self):
        return self.chanel_names_in_dewesoft
    
class TubeSetupModel:
    def __init__(self, data_store : DataStore):
        self.data = {
            'mic_spacing': '',
            'mic1_sample': '',
            'mic2_sample': '',
            'tube_diameter': ''
        }
        self.data_store = data_store

    def set_data(self, data):
        self.data_store.add_tube_measurements(data['mic_spacing'], data['tube_diameter'], data['mic1_sample'], data['mic2_sample'])

class ResultsModel:
    def __init__(self, data_store : DataStore):
        self.data_store = data_store
        self.processed_measurements = []  # List of processed measurements
        self.selected_metrics = set()  # Set of selected evaluation metrics

    def get_processed_measurements(self):
        return self.data_store.get_results()

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
    
    def generate_plot(self, metric, measument_name):
        measurement_data = self.get_measurement_by_name(measument_name, metric)

        if metric == "Calibration Function":
            interchanged_measurement = self.get_measurement_by_name("TestAbsorcao_MicTrocado", metric)
            fig = self.calibration_function_graph(mic_normal=measurement_data, mic_changed=interchanged_measurement)
            return fig
        if metric == "Absorption Coefficient":
            interchanged_measurement = self.get_measurement_by_name("TestAbsorcao_MicTrocado", metric)
            fig = self.absorption_coef_graph(measurement_data=measurement_data,mic_changed=interchanged_measurement)
            return fig
            
        if metric == "Fourier Transform":
            try:
                signal_1 = np.asarray(measurement_data['AI A-1']).flatten()
                time = np.asarray(measurement_data['Time']).flatten()
            except:
                channel_names = self.data_store.get_channel_names()
                channel_: str
                for name in channel_names:
                    if 'time' not in name.lower():
                        channel_ = name
                        break
                signal_1 = measurement_data[channel_]
                time = measurement_data['time']
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(time, signal_1, label="Original Signal", color="blue")


            # Compute sampling interval and length
            dt = 1 / measurement_data['SampleRate']
            fs = 1 / dt

            # STFT parameters
            nperseg = 10000
            noverlap = int(nperseg * 2 / 3)  # 66.67% overlap

            # Perform STFT
            f, t_spec, Zxx = stft(signal_1, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)

            # Only keep frequencies from 0 to 1600 Hz
            mask = (f >= 0) & (f <= 1600)
            f = f[mask]
            Zxx = Zxx[mask, :]

            # Convert to amplitude (Pa)
            amplitude = np.abs(Zxx)

            # Mean amplitude across time for a 1D FFT-like plot
            mean_amplitude = np.mean(amplitude, axis=1)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(f, mean_amplitude, label='Mean STFT Amplitude (Pa)', color='green')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude (Pa)")
            ax.set_title("Short-Time FFT (0 to 1600 Hz)")
            ax.grid()
            ax.legend()

            return fig
        
        if metric == "Original Signal":
            try:
                signal_1 = measurement_data['AI A-1']
                signal_2 = measurement_data['AI A-2']
                time = measurement_data['Time']
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(time, signal_1, label="Original Signal", color="blue")
                ax.plot(time, signal_2, label="Signal 2 (AI A-2)", color="orange")
            except:
                channel_names = self.data_store.get_channel_names()
                channel_: str
                for name in channel_names:
                    if 'time' not in name.lower():
                        channel_ = name
                        break
                signal_1 = measurement_data[channel_]
                time = measurement_data['time']
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(time, signal_1, label="Original Signal", color="blue")


            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Magnitude')
            ax.set_title('Original Signal')
            ax.grid()

            return fig  # Return the figure
        
        if metric == "Transfer Function":
            calc = CalculationISO()
            cali = CalibrationCalculations()

            mic_changed = self.get_measurement_by_name("TestAbsorcao_MicTrocado", metric)

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
            
            freq, H12, H12m, H12p = calc.process_time_domain_data(
                measurement_data['AI A-1'], 
                measurement_data['AI A-2'], 
                measurement_data["SampleRate"]
            )

            print(f"Frequencies: {freq}")
            
            _, H21, H21m, H21p = calc.process_time_domain_data(
                mic_changed_adjusted['AI A-1'], 
                mic_changed_adjusted['AI A-2'], 
                mic_changed_adjusted["SampleRate"]
            )

            Hc = cali.get_calibration_factor_switching_method(H12_normal=H12, H12_switched=H21)

            print(f"Calibration factor {Hc}")

            #H12 = cali.apply_calibration_factor(H12,Hc)

            blockSize = 10000
            overlap = 0.75 # 0.75 significa 75%
            fmax = 1600 # Frequencia maxima do grafico. A frequencia de amostragem é definida pelo dt.

            time = np.asarray(measurement_data['Time']).flatten()
            p001 = np.asarray(measurement_data['AI A-1']).flatten()
            p002 = np.asarray(measurement_data['AI A-2']).flatten()

            nt = int(len(p001))

            dt = time[1]

            fsampling = 1/dt

            w = np.hanning(blockSize)
            blockShift = int(blockSize*(1-overlap))

            SFT = ShortTimeFFT(w, hop=blockShift, fs=fsampling, scale_to='magnitude')

            Sx001 = SFT.stft(p001)  # perform the STFT
            #Sx001 = 2*Sx001 # Verificar se isso é necessário.

            Sx002 = SFT.stft(p002)  # perform the STFT
            #Sx002 = 2*Sx002 # Verificar se isso é necessário.

            Sx001_mag   = np.abs(Sx001)
            Sx001_phase = np.angle(Sx001)
            Sx001_real  = Sx001.real
            Sx001_imag  = Sx001.imag

            Sx002_mag   = np.abs(Sx002)
            Sx002_phase = np.angle(Sx002)
            Sx002_real  = Sx002.real
            Sx002_imag  = Sx002.imag


            nBlock = np.size(Sx001_mag,1)

            blockTime = blockSize*dt
            df = 1/blockTime
            nf = int(np.floor(blockSize*0.5))+1

            freq = np.linspace(0,fsampling*0.5,nf)
            t_lo, t_hi = SFT.extent(nt)[:2]
            t1 = SFT.lower_border_end[0] * SFT.T
            t2 = SFT.upper_border_begin(nt)[0] * SFT.T

            timeBlock = np.linspace(t_lo,t_hi,nBlock)

            Sx001_mag_average   = np.zeros(shape=(nf,))
            Sx001_phase_average = np.zeros(shape=(nf,))
            Sx001_real_average  = np.zeros(shape=(nf,))
            Sx001_imag_average  = np.zeros(shape=(nf,))

            Sx002_mag_average   = np.zeros(shape=(nf,))
            Sx002_phase_average = np.zeros(shape=(nf,))
            Sx002_real_average  = np.zeros(shape=(nf,))
            Sx002_imag_average  = np.zeros(shape=(nf,))


            for i in range(0,nf):
                Sx001_mag_average[i]=np.average(Sx001_mag[i,:])
                Sx001_phase_average[i]=np.average(Sx001_phase[i,:])
                Sx001_real_average[i]=np.average(Sx001_real[i,:])
                Sx001_imag_average[i]=np.average(Sx001_imag[i,:])
                Sx002_mag_average[i]=np.average(Sx002_mag[i,:])
                Sx002_phase_average[i]=np.average(Sx002_phase[i,:])
                Sx002_real_average[i]=np.average(Sx002_real[i,:])
                Sx002_imag_average[i]=np.average(Sx002_imag[i,:])


            H12_real  = np.zeros(shape=(nf,))
            H12_imag  = np.zeros(shape=(nf,))
            H12_mag  = np.zeros(shape=(nf,))
            H12_phase  = np.zeros(shape=(nf,))
            for i in range(0,nf):
                x2 = Sx001_real_average[i]
                x1 = Sx002_real_average[i]
                y2 = Sx001_imag_average[i]
                y1 = Sx002_imag_average[i]
                H12_real[i] = (x1*x2+y1*y2)/(x2*x2+y2*y2)
                H12_imag[i] = (y1*x2-x1*y2)/(x2*x2+y2*y2)
                H12_mag[i] = np.sqrt(H12_real[i]*H12_real[i] + H12_imag[i]*H12_imag[i])
                H12_phase[i] = np.arctan2(H12_imag[i],H12_real[i])

            # Plot
            fig, ax = plt.subplots(figsize=(12, 5))
            #ax.plot(freq, np.angle(H12), label='phase of H1', color='green')
            ax.semilogy(freq, H12m,label='Tranfer Function H1', color='blue')
            ax.semilogy(freq,H12_mag,'-',label='H1 Celso', color='orange')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("")
            ax.set_title("H1 - calibrated")
            ax.grid()
            ax.legend()

            return fig

    
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
        
        freq, H12, H12m, H12p = calc.process_time_domain_data(
            measurement_data_adjusted['AI A-1'], 
            measurement_data_adjusted['AI A-2'], 
            measurement_data_adjusted["SampleRate"]
        )

        print(f"Frequencies: {freq}")
        
        _, H21, H21m, H21p = calc.process_time_domain_data(
            mic_changed_adjusted['AI A-1'], 
            mic_changed_adjusted['AI A-2'], 
            mic_changed_adjusted["SampleRate"]
        )

        Hc = cali.get_calibration_factor_switching_method(H12_normal=H12, H12_switched=H21)

        print(f"Calibration factor {Hc}")

        H12c = cali.apply_calibration_factor(H12,Hc)

        print(f"H12 after calibration: {H12}")

        alphac = calc.calculate_absorption_coeficent(freq, H12c, 0.1, 0.05)

        f_min, f_max = calc.calculate_valid_frequency_range(0.1, 0.05, measurement_data_adjusted["SampleRate"])

        print(f"Fmax and Fmin : {f_max} {f_min}")

        self.data_store.add_absorption_coef(coef=(freq, alphac), name="Absorption Coef")

        fig, ax = calc.plot_results(freq, alphac)
        print(fig)
        return fig
    
    def calibration_function_graph(self, mic_normal, mic_changed):
        calc = CalculationISO()
        cali = CalibrationCalculations()

        # Ajustar os dados brutos para o mesmo tamanho
        min_length_ch1 = min(len(mic_normal['AI A-1']), len(mic_changed['AI A-1']))
        min_length_ch2 = min(len(mic_normal['AI A-2']), len(mic_changed['AI A-2']))
        
        # Garantir que os dois canais tenham o mesmo comprimento
        min_length = min(min_length_ch1, min_length_ch2)
        
        # Cortar os dados brutos
        mic_normal_adjusted = {
            'AI A-1': mic_normal['AI A-1'][:min_length],
            'AI A-2': mic_normal['AI A-2'][:min_length],
            'SampleRate': mic_normal['SampleRate']
        }
        
        mic_changed_adjusted = {
            'AI A-1': mic_changed['AI A-1'][:min_length],
            'AI A-2': mic_changed['AI A-2'][:min_length],
            'SampleRate': mic_changed['SampleRate']
        }
        
        freq, H12, H12m, H12p = calc.process_time_domain_data(
            mic_normal_adjusted['AI A-1'], 
            mic_normal_adjusted['AI A-2'], 
            mic_normal_adjusted["SampleRate"]
        )

        print(f"Frequencies: {freq}")
        
        _, H21, H21m, H21p = calc.process_time_domain_data(
            mic_changed_adjusted['AI A-1'], 
            mic_changed_adjusted['AI A-2'], 
            mic_changed_adjusted["SampleRate"]
        )

        Hc = cali.get_calibration_factor_switching_method(H12_normal=H12, H12_switched=H21)

        fig, ax = calc.plot_calibration_function(frequencies=freq, calibration_function=Hc)

        return fig
        

class MeasurementModel:
    def __init__(self, data_store : DataStore):
        self.data_store = data_store

    def get_samples(self):
        """Get all sample names from the data store."""
        return list(self.data_store.samples.keys())

    def get_measurement_results(self):
        return self.data_store.get_results()

    def add_measurement_result(self, result, name):
        """Adds a measurement result to the results dict."""
        self.data_store.add_result(result, name)

    def save_channel_names(self, names: list[str]):
        self.data_store.chanel_names_in_dewesoft = names

    def get_channel_names(self):
        return self.data_store.chanel_names_in_dewesoft

class SamplesModel:
    def __init__(self,data_store : DataStore):
        self.data_store = data_store

    def get_samples(self):
        """Get all sample names from the data store."""
        return list(self.data_store.samples.keys())

    def save_sample(self, sample):
        sample_name = sample["sample_name"]
        self.data_store.add_sample(sample=sample, sample_name=sample_name)

class WarningsModel:
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        
    def save_warning_settings(self, snr, auto_spec, calib):
        """Save warning settings to the data store."""
        try:
            self.data_store.add_warnings(snr=snr, auto_spec=auto_spec, calib=calib)
            return True
        except Exception as e:
            print(f"Error saving warning settings: {e}")
            return False
    
    def get_warning_settings(self):
        """Get current warning settings from the data store."""
        try:
            return self.data_store.get_warnings()
        except Exception as e:
            print(f"Error getting warning settings: {e}")
            return {
                'snr': '',
                'auto_spec': '',
                'calib': ''
            }

class ProcessingModel:
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.next_id = 1  # For generating unique IDs for processed results
    
    def get_processed_measurements(self):
        """Get list of processed measurement results.
        
        Returns:
            list: Names of processed measurements
        """
        return list(self.data_store.get_all_post_processed().keys())
    
    def get_processed_data(self, name):
        """Get processed data by name.
        
        Args:
            name (str): Name of the processed result
            
        Returns:
            dict: Processed data or None if not found
        """
        return self.data_store.get_post_processed(name)
    
    def get_available_measurements(self):
        return self.data_store.get_results().keys()
    
    def process_measurements(self, selected_measurements, operations):
        """Process selected measurements with specified operations.
        
        Args:
            selected_measurements (list): List of measurement names to process
            operations (list): List of operation dictionaries
            
        Returns:
            str: Name of the resulting processed measurement
            
        Raises:
            ValueError: If no measurements selected or invalid operation
            Exception: For any processing errors
        """
        if not selected_measurements:
            raise ValueError("No measurements selected for processing")
        
        # Get data for selected measurements
        measurement_data = [name for name in selected_measurements if name in self.data_store.get_results()]
        
        if not measurement_data:
            raise ValueError("None of the selected measurements are available")
        
        # Apply operations in sequence
        result_data = measurement_data.copy()
        operation_summary = []
        
        for operation in operations:
            op_type = operation['type']
            
            if op_type == 'average':
                result_data = self._average_data(result_data, operation['method'])
                operation_summary.append(f"Averaged ({operation['method']})")
            
            elif op_type == 'combine':
                result_data = self._combine_data(result_data, operation['method'])
                operation_summary.append(f"Combined ({operation['method']})")
            
            elif op_type == 'extract':
                result_data = self._extract_third_octave(
                    result_data, operation['min_freq'], operation['max_freq'])
                operation_summary.append(f"Extracted third-octave ({operation['min_freq']}-{operation['max_freq']} Hz)")
            
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
        
        # Create a name for the processed result
        result_name = f"Processed_{self.next_id}_{'-'.join(operation_summary)}"
        self.next_id += 1
        
        # Store the processed result
        self.data_store.add_post_processed(result_data, result_name) 
        
        return result_name
    
    def _average_data(self, data_list, method='Arithmetic Mean'):
        """Average multiple measurements.
        
        Args:
            data_list (list): List of measurement data dictionaries
            method (str): Averaging method
            
        Returns:
            dict: Averaged data
        """
        if not data_list:
            raise ValueError("No data to average")
        
        # For simplicity in this example, assuming all data has the same format
        # In a real implementation, you'd need to handle different data formats
        
        result = {}
        
        # Get frequency points (assuming all measurements have the same frequency points)
        if isinstance(data_list[0], dict) and 'frequency' in data_list[0]:
            result['frequency'] = data_list[0]['frequency'].copy()
        
        # Different averaging methods
        if method == 'Arithmetic Mean':
            # Average amplitude values at each frequency point
            if isinstance(data_list[0], dict) and 'amplitude' in data_list[0]:
                amplitudes = [d['amplitude'] for d in data_list]
                import numpy as np
                result['amplitude'] = np.mean(amplitudes, axis=0).tolist()
        
        elif method == 'Geometric Mean':
            # Geometric mean of amplitude values
            if isinstance(data_list[0], dict) and 'amplitude' in data_list[0]:
                amplitudes = [d['amplitude'] for d in data_list]
                import numpy as np
                result['amplitude'] = np.exp(np.mean(np.log(amplitudes), axis=0)).tolist()
        
        elif method == 'Weighted Average':
            # Example of weighted average (equal weights in this case)
            if isinstance(data_list[0], dict) and 'amplitude' in data_list[0]:
                amplitudes = [d['amplitude'] for d in data_list]
                weights = [1/len(amplitudes)] * len(amplitudes)
                import numpy as np
                result['amplitude'] = np.average(amplitudes, axis=0, weights=weights).tolist()
        
        # Include metadata
        result['processing'] = {
            'type': 'average',
            'method': method,
            'source_count': len(data_list)
        }
        
        return result
    
    def _combine_data(self, data_list, method='Sequential'):
        """Combine multiple measurements.
        
        Args:
            data_list (list): List of measurement data dictionaries
            method (str): Combination method
            
        Returns:
            dict: Combined data
        """
        if not data_list:
            raise ValueError("No data to combine")
        
        result = {}
        
        # Different combination methods
        if method == 'Sequential':
            # Concatenate measurements sequentially
            import numpy as np
            result['frequency'] = []
            result['amplitude'] = []
            
            for data in data_list:
                if isinstance(data, dict) and 'frequency' in data and 'amplitude' in data:
                    result['frequency'].extend(data['frequency'])
                    result['amplitude'].extend(data['amplitude'])
        
        elif method == 'Parallel':
            # Merge measurements (assuming different frequency ranges)
            import numpy as np
            all_freqs = []
            all_amps = []
            
            for data in data_list:
                if isinstance(data, dict) and 'frequency' in data and 'amplitude' in data:
                    all_freqs.extend(data['frequency'])
                    all_amps.extend(data['amplitude'])
            
            # Sort by frequency
            freq_amp = sorted(zip(all_freqs, all_amps), key=lambda x: x[0])
            result['frequency'] = [fa[0] for fa in freq_amp]
            result['amplitude'] = [fa[1] for fa in freq_amp]
        
        elif method == 'Custom':
            # Example of a custom combination (averaging within frequency bands)
            # This is a simplified example
            result = self._average_data(data_list, 'Arithmetic Mean')
        
        # Include metadata
        result['processing'] = {
            'type': 'combine',
            'method': method,
            'source_count': len(data_list)
        }
        
        return result
    
    def _extract_third_octave(self, data_list, min_freq=125, max_freq=4000):
        """Extract third-octave bands from data.
        
        Args:
            data_list (list): List of measurement data dictionaries
            min_freq (int): Minimum frequency for extraction
            max_freq (int): Maximum frequency for extraction
            
        Returns:
            dict: Data with third-octave bands
        """
        # Simple implementation - in a real scenario this would be more complex
        import numpy as np
        
        # Standard third-octave band center frequencies
        third_octave_centers = [
            16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 
            200, 250, 315, 400, 500, 630, 800, 1000, 1250, 
            1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000
        ]
        
        # Filter frequencies within range
        centers = [f for f in third_octave_centers if min_freq <= f <= max_freq]
        
        result = {
            'frequency': centers,
            'amplitude': [],
            'processing': {
                'type': 'third_octave',
                'min_freq': min_freq,
                'max_freq': max_freq
            }
        }
        
        # Extract values for each band from each dataset and average
        for center in centers:
            band_values = []
            
            for data in data_list:
                if isinstance(data, dict) and 'frequency' in data and 'amplitude' in data:
                    # Calculate band limits (third-octave bandwidth)
                    lower = center / 2**(1/6)
                    upper = center * 2**(1/6)
                    
                    # Find values within this band
                    band_indices = [i for i, f in enumerate(data['frequency']) 
                                     if lower <= f <= upper]
                    
                    if band_indices:
                        # Average the values in this band
                        band_avg = np.mean([data['amplitude'][i] for i in band_indices])
                        band_values.append(band_avg)
            
            # Average across all datasets for this band
            if band_values:
                result['amplitude'].append(np.mean(band_values))
            else:
                result['amplitude'].append(0)  # No data for this band
        
        return result
    
    def clear_measurement(self, name):
        """Remove a measurement from the model."""
        if name in self.raw_measurements:
            del self.raw_measurements[name]
        if name in self.processed_measurements:
            del self.processed_measurements[name]
    
    def import_measurements_from_file(self, file_path):
        """Import measurements from a file.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            list: Names of imported measurements
        """
        try:
            # Simple implementation - in a real app, this would parse actual files
            import json
            import os
            import random
            
            # For demo purposes, generate synthetic data if file doesn't exist
            if not os.path.exists(file_path):
                # Create some synthetic data for demonstration
                return self._generate_synthetic_data(3)
            
            # In real implementation, parse the file
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Process the data and add measurements
            imported = []
            for name, measurement_data in data.items():
                self.add_measurement(name, measurement_data)
                imported.append(name)
                
            return imported
            
        except Exception as e:
            print(f"Error importing measurements: {e}")
            return []
    
    def _generate_synthetic_data(self, count=3):
        """Generate synthetic data for testing.
        
        Args:
            count (int): Number of measurements to generate
            
        Returns:
            list: Names of generated measurements
        """
        import numpy as np
        import random
        
        names = []
        
        for i in range(1, count + 1):
            name = f"Sample_{i}_{random.choice(['Low', 'Mid', 'High'])}_{random.randint(1, 100)}"
            
            # Generate frequency points
            frequencies = np.logspace(1, 4, 100).tolist()  # 10 Hz to 10 kHz
            
            # Generate amplitude data (with some randomness)
            base_curve = -20 * np.log10(frequencies)  # Basic decay curve
            noise = np.random.normal(0, 5, len(frequencies))  # Add noise
            amplitude = base_curve + noise
            
            # Normalize to reasonable range
            amplitude = amplitude - np.min(amplitude)
            amplitude = amplitude / np.max(amplitude) * 70
            
            # Create measurement data
            data = {
                'frequency': frequencies,
                'amplitude': amplitude.tolist(),
                'metadata': {
                    'date': '2025-04-17',
                    'sample_type': f"Type {random.randint(1, 5)}",
                    'thickness': random.uniform(10, 50)
                }
            }
            
            self.add_measurement(name, data)
            names.append(name)
        
        return names
    
    def export_processed_data(self, name, file_path):
        """Export a processed measurement to a file.
        
        Args:
            name (str): Name of the processed measurement
            file_path (str): Path to save the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name not in self.processed_measurements:
            return False
        
        try:
            # Simple implementation - in a real app, this would create actual files
            import json
            import os
            
            data = self.processed_measurements[name]
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
        
class ReportModel:
    """Model for handling report data and generation."""
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.report_data = {
            'client_info': {
                'name': '',
                'company': '',
                'address': '',
                'purchase_order': '',
                'test_date': None,
                'report_number': '',
                'test_requester': '',
                'test_executor': '',
                'test_supervisor': ''
            },
            'sample_info': {
                'product_name': '',
                'manufacturer': '',
                'description': '',
                'layers': '',
                'conditions': '',
                'mounting': ''
            },
            'test_conditions': None,  # Will be populated from data_store
            'equipment': [
                {
                    'name': "DewesoftX",
                    'serial': "2477213 / PULSE N.2"
                },
                {
                    'name': "Impedance Tube Type 4206",
                    'serial': "2681869"
                },
                {
                    'name': "Desktop PC with Software and Peripheral Equipment",
                    'serial': "21329"
                },
                {
                    'name': "Two ¼ Condenser Microphones Type 4187",
                    'serial': "2677390 & 2677391"
                },
                {
                    'name': "Power Amplifier",
                    'serial': "129003364"
                }
            ],
            'tube_setup': None,  # Will be populated from data_store
            'measurements': None,  # Will be populated from data_store
            'results': None  # Will be populated from data_store
        }

    def update_client_info(self, client_info: dict):
        """Update client information for the report."""
        self.report_data['client_info'].update(client_info)

    def update_sample_info(self, sample_info: dict):
        """Update sample information for the report."""
        self.report_data['sample_info'].update(sample_info)

    def get_client_info(self) -> dict:
        """Get current client information."""
        return self.report_data['client_info']

    def get_sample_info(self) -> dict:
        """Get current sample information."""
        return self.report_data['sample_info']

    def validate_report_data(self) -> tuple[bool, list[str]]:
        """Validate report data before generation.
        
        Returns:
            tuple: (is_valid, list of error messages)
        """
        errors = []
        client_info = self.report_data['client_info']
        
        # Required fields
        required_fields = {
            'name': 'Client Name',
            'report_number': 'Report Number',
            'test_date': 'Test Date',
            'test_executor': 'Test Executor'
        }
        
        for field, display_name in required_fields.items():
            if not client_info.get(field):
                errors.append(f"{display_name} is required")
        
        return len(errors) == 0, errors

    def load_test_conditions(self):
        """Load test conditions from data store."""
        print("Loading test conditions")
        self.report_data['test_conditions'] = self.data_store.get_test_conditions()

    def load_tube_setup(self):
        """Load tube setup from data store."""
        print("Loading tube setup")
        self.report_data['tube_setup'] = self.data_store.get_tube_measurements()

    def load_measurements(self):
        """Load measurement results from data store."""
        print("Loading measurements")
        self.report_data['measurements'] = self.data_store.get_results()
        print(self.report_data['measurements'])

    def load_results(self):
        """Load processed results from data store."""
        print("Loading results")
        self.report_data['results'] = self.data_store.get_all_post_processed()
        print(self.report_data['results'])

    def load_absorption_data(self):
        """Load absorption data from data store."""
        print("Loading absorption data")
        self.report_data['absorption_data'] = self.data_store.get_absorption_coef("Absorption Coef")
        print(self.report_data['absorption_data'])

    def get_absorption_data(self):
        """Get absorption coefficient data for different tube diameters at 250Hz intervals."""
        if not self.report_data['measurements'] or not self.report_data['absorption_data']:
            print("No measurements or absorption data found")
            return []

        results = []
        absorption_data = self.report_data['absorption_data']
        
        print("Frequency array:", absorption_data[0])
        print("Absorption array:", absorption_data[1])
        
        # Extract frequency and absorption arrays
        frequencies = absorption_data[0]
        absorptions = absorption_data[1]
        
        # Define the frequency intervals (250Hz steps)
        target_frequencies = [500, 750, 1000,1250, 1500, 1750, 2000]
        
        # Find the closest frequency in the data for each target frequency
        for target_freq in target_frequencies:
            # Find the closest frequency in the data
            closest_idx = min(range(len(frequencies)), 
                            key=lambda i: abs(frequencies[i] - target_freq))
            actual_freq = frequencies[closest_idx]
            absorption_value = absorptions[closest_idx]
            
            if abs(actual_freq - target_freq) <= 50:
                alpha_100mm = absorption_value
                alpha_29mm = absorption_value  # Placeholder - should be separate data
                combined = (alpha_100mm + alpha_29mm) / 2
                
                results.append((target_freq, alpha_100mm, alpha_29mm, combined))
                print(f"Added data for {target_freq}Hz: {absorption_value:.3f}")
            else:
                print(f"No data found close to {target_freq}Hz (closest: {actual_freq}Hz)")

        return sorted(results, key=lambda x: x[0])  # Sort by frequency

    def get_report_metadata(self):
        """Get metadata for report generation."""
        return {
            'client_info': self.report_data['client_info'],
            'test_conditions': self.report_data['test_conditions'],
            'equipment': self.report_data['equipment']
        }

    def refresh_data(self):
        """Refresh all data from data store."""
        print("Refreshing data")
        self.load_test_conditions()
        self.load_tube_setup()
        self.load_measurements()
        self.load_results()
        self.load_absorption_data()

class TestConditionsModel:
    def __init__(self, data_store : DataStore):
        self._data = {
            "temperature": {"value": None, "unit": None},
            "pressure": {"value": None, "unit": None},
            "humidity": {"value": None, "unit": "%"}
        }
        self.data_store = data_store

    def set_temperature(self, value: float, unit: str):
        self._data["temperature"] = {"value": value, "unit": unit}

    def set_pressure(self, value: float, unit: str):
        self._data["pressure"] = {"value": value, "unit": unit}

    def set_humidity(self, value: float):
        self._data["humidity"] = {"value": value, "unit": "%"}

    def save_all(self):
        self.data_store.add_test_conditions(self._data["temperature"],self._data["pressure"],self._data["humidity"])

    def get_data(self) -> dict:
        return self._data

class DocumentationModel:
    """Model for managing documentation content and structure."""
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.documentation = {
            'tube_setup': {
                'title': 'Impedance Tube Setup Guide',
                'content': '''
                The impedance tube setup requires careful attention to the following aspects:
                
                1. Microphone Spacing
                   - Must be appropriate for the frequency range of interest
                   - Affects the working frequency range
                   - Should be calibrated and verified regularly
                
                2. Tube Diameter
                   - Determines the upper frequency limit
                   - Must be appropriate for the sample size
                   - Affects the plane wave propagation
                
                3. Sample Mounting
                   - Must be secure and airtight
                   - Sample should fit snugly in the tube
                   - Avoid air gaps between sample and tube wall
                
                4. Equipment Verification
                   - Check microphone calibration
                   - Verify signal generator functionality
                   - Ensure proper amplifier settings
                
                5. Environmental Conditions
                   - Monitor temperature and humidity
                   - Keep conditions stable during measurements
                   - Document all environmental parameters
                '''
            },
            'measurement_procedure': {
                'title': 'Measurement Procedure Guide',
                'content': '''
                Follow these steps for accurate measurements:
                
                1. Pre-measurement Checks
                   - Verify all equipment is properly connected
                   - Check microphone positions
                   - Ensure sample is properly mounted
                
                2. Signal Setup
                   - Set appropriate frequency range
                   - Adjust signal amplitude
                   - Check for signal distortion
                
                3. Data Collection
                   - Record background noise levels
                   - Perform calibration measurements
                   - Take multiple measurements for averaging
                
                4. Quality Control
                   - Monitor signal-to-noise ratio
                   - Check for measurement consistency
                   - Verify data validity
                
                5. Data Storage
                   - Save raw measurement data
                   - Document measurement conditions
                   - Backup all results
                '''
            },
            'data_processing': {
                'title': 'Data Processing Guidelines',
                'content': '''
                Data processing involves several important steps:
                
                1. Raw Data Analysis
                   - Import measurement data
                   - Check data integrity
                   - Apply calibration corrections
                
                2. Signal Processing
                   - Apply appropriate windowing
                   - Perform FFT analysis
                   - Calculate transfer functions
                
                3. Acoustic Parameters
                   - Calculate absorption coefficients
                   - Determine impedance values
                   - Evaluate measurement uncertainty
                
                4. Result Validation
                   - Compare with expected values
                   - Check for physical consistency
                   - Identify anomalies
                
                5. Documentation
                   - Generate comprehensive reports
                   - Include all relevant metadata
                   - Archive processed results
                '''
            },
            'troubleshooting': {
                'title': 'Troubleshooting Guide',
                'content': '''
                Common issues and their solutions:
                
                1. Poor Signal Quality
                   - Check cable connections
                   - Verify microphone power supply
                   - Inspect for damaged components
                
                2. Inconsistent Results
                   - Verify sample mounting
                   - Check environmental stability
                   - Review calibration status
                
                3. System Errors
                   - Restart measurement software
                   - Check system configuration
                   - Verify hardware compatibility
                
                4. Data Issues
                   - Backup corrupted files
                   - Verify storage space
                   - Check file permissions
                
                5. Calibration Problems
                   - Recalibrate microphones
                   - Check reference standards
                   - Document calibration history
                '''
            },
            'maintenance': {
                'title': 'Equipment Maintenance',
                'content': '''
                Regular maintenance procedures:
                
                1. Daily Checks
                   - Clean tube interior
                   - Inspect microphones
                   - Check cable connections
                
                2. Weekly Maintenance
                   - Calibrate microphones
                   - Verify system performance
                   - Update software if needed
                
                3. Monthly Tasks
                   - Deep clean all components
                   - Check for wear and tear
                   - Update documentation
                
                4. Quarterly Service
                   - Full system calibration
                   - Hardware inspection
                   - Software updates
                
                5. Annual Maintenance
                   - Professional servicing
                   - Replace worn components
                   - Validate system accuracy
                '''
            }
        }

    def get_documentation_section(self, section_name: str) -> dict:
        """
        Retrieve a specific documentation section.
        
        Args:
            section_name (str): Name of the section to retrieve
            
        Returns:
            dict: Section data containing title and content, or None if not found
        """
        return self.documentation.get(section_name)

    def get_all_sections(self) -> dict:
        """
        Get all documentation sections.
        
        Returns:
            dict: All documentation sections
        """
        return self.documentation

    def update_section(self, section_name: str, title: str, content: str) -> bool:
        """
        Update a documentation section.
        
        Args:
            section_name (str): Name of the section to update
            title (str): New title for the section
            content (str): New content for the section
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if section_name in self.documentation:
            self.documentation[section_name] = {
                'title': title,
                'content': content
            }
            return True
        return False

    def add_section(self, section_name: str, title: str, content: str) -> bool:
        """
        Add a new documentation section.
        
        Args:
            section_name (str): Name for the new section
            title (str): Title for the new section
            content (str): Content for the new section
            
        Returns:
            bool: True if addition successful, False if section already exists
        """
        if section_name not in self.documentation:
            self.documentation[section_name] = {
                'title': title,
                'content': content
            }
            return True
        return False

    def delete_section(self, section_name: str) -> bool:
        """
        Delete a documentation section.
        
        Args:
            section_name (str): Name of the section to delete
            
        Returns:
            bool: True if deletion successful, False if section not found
        """
        if section_name in self.documentation:
            del self.documentation[section_name]
            return True
        return False

    def search_documentation(self, query: str) -> list:
        """
        Search through documentation content.
        
        Args:
            query (str): Search query string
            
        Returns:
            list: List of matching section names and titles
        """
        results = []
        query = query.lower()
        for section_name, section_data in self.documentation.items():
            if (query in section_name.lower() or
                query in section_data['title'].lower() or
                query in section_data['content'].lower()):
                results.append({
                    'section_name': section_name,
                    'title': section_data['title']
                })
        return results

def run():
    dreader = DWDataReader()
    dreader.open_data_file("TestFundo")
    info = dreader.get_measurement_info()
    print(info.sample_rate)
    data = dreader.get_measurements_as_dataframe()
    print(data)



if __name__ == "__main__":
    run()
