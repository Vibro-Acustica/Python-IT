from win32com.client import Dispatch
import time
import sys

# create DCOM object
dw = Dispatch("Dewesoft.App")

print("Initializing Dewesoft... ", end = "")
sys.stdout.flush()
dw.Init()
dw.Enabled = 1
dw.Visible = 1
print('done.')

# set window dimensions
dw.Top = 0
dw.Left = 0
dw.Width = 1024
dw.Height = 768

setup = "C:\\Users\\jvv20\\Vibra\\DeweSoftData\\Setups\\test.dxs"

dw.LoadSetup(setup)

new_file_name = setup.split('\\')[-1].replace('.dxs', "res")+'.dxd'
# set sample rate to 10 KHz
dw.MeasureSampleRate = 10000

# start measure mode and wait a little
delay = 5
print("Running measure mode for %d seconds..." % delay)
dw.Start()
dw.StartStoring(new_file_name)
time.sleep(delay)
dw.Stop()

# close dewesoft
print("Closing Dewesoft... ", end = "")
sys.stdout.flush()
dw = None
print("done.")
sys.stdout.flush()
