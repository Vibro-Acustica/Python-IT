import time
import sys
from win32com.client import Dispatch
from DataReader import DWDataReader

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

def run():
    dewesoft = Dewesoft()
    dewesoft.set_sample_rate(1000) 
    dewesoft.set_dimensions(800, 600)
    dewesoft.load_setup("C:\\Users\\jvv20\\Vibra\\DeweSoftData\\Setups\\test.dxs")
    
    dewesoft.measure(2, "first")
    #dewesoft.measure(2, "second")
    
    dewesoft.close()

    dreader = DWDataReader()
    dreader.open_data_file("first")
    info = dreader.get_measurement_info()
    print(info)
    data = dreader.get_measurements_as_dataframe()
    print(data)


if __name__ == "__main__":
    run()
