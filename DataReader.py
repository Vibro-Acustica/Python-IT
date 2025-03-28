import pandas as pd
from DWDataReaderHeader import *
from ctypes import *
import _ctypes

class DWDataReader:
    def __init__(self, dll_path='.\\DWDataReaderLib64.dll', file_paths = "C:\\Users\\jvv20\\Vibra\\DeweSoftData\\Data\\"):
        self.lib = cdll.LoadLibrary(dll_path)
        self.file_paths = file_paths
        if self.lib.DWInit() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWInit() failed")
        self.add_reader()
        self.ensure_everything_is_ok()

    def raise_error(self, message):
        raise Exception(f"DWDataReader: {message}")

    def add_reader(self):
        if self.lib.DWAddReader() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWAddReader() failed")

    def open_data_file(self, file_name):
        full_name = self.file_paths + file_name + ".dxd"
        file_name_c = c_char_p(full_name.encode())
        self.file_info = DWFileInfo(0, 0, 0)
        if self.lib.DWOpenDataFile(file_name_c, c_void_p(addressof(self.file_info))) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWOpenDataFile() failed")

    def get_measurement_info(self):
        measurement_info = DWMeasurementInfo(0, 0, 0, 0)
        if self.lib.DWGetMeasurementInfo(c_void_p(addressof(measurement_info))) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWGetMeasurementInfo() failed")
        return measurement_info

    def get_channel_list(self):
        num_channels = self.lib.DWGetChannelListCount()
        if num_channels == -1:
            self.raise_error("DWGetChannelListCount() failed")
        ch_list = (DWChannel * num_channels)()
        if self.lib.DWGetChannelList(byref(ch_list)) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWGetChannelList() failed")
        return ch_list

    def get_scaled_samples(self, channel_index):
        dw_ch_index = c_int(channel_index)
        sample_cnt = self.lib.DWGetScaledSamplesCount(dw_ch_index)
        if sample_cnt < 0:
            self.raise_error("DWGetScaledSamplesCount() failed")
        data = create_string_buffer(DOUBLE_SIZE * sample_cnt)
        time_stamp = create_string_buffer(DOUBLE_SIZE * sample_cnt)
        p_data = cast(data, POINTER(c_double))
        p_time_stamp = cast(time_stamp, POINTER(c_double))
        if self.lib.DWGetScaledSamples(dw_ch_index, c_int64(0), sample_cnt, p_data, p_time_stamp) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWGetScaledSamples() failed")
        return [(p_time_stamp[i], p_data[i]) for i in range(sample_cnt)]
    
    def ensure_everything_is_ok(self):
        if self.lib.DWGetNumReaders(byref(c_int())) != DWStatus.DWSTAT_OK.value:
            self.raise_error("Data readers are not initialized correctly")
        print("All systems are operational.")
    
    def get_measurements_as_dataframe(self):
        ch_list = self.get_channel_list()
        data = {}
        for ch in ch_list:
            samples = self.get_scaled_samples(ch.index)
            data[ch.name.decode()] = [val[1] for val in samples]
        return pd.DataFrame(data)
    
    def close(self):
        if self.lib.DWCloseDataFile() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWCloseDataFile() failed")
        if self.lib.DWDeInit() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWDeInit() failed")
        _ctypes.FreeLibrary(self.lib._handle)
        del self.lib
