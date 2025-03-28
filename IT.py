import time
import sys
from win32com.client import Dispatch
from DataReader import DWDataReader

class DewesoftController:
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
    dewesoft = DewesoftController()
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
