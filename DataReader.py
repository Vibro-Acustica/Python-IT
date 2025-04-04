import pandas as pd
import numpy as np
from DWDataReaderHeader import *
from ctypes import *
import _ctypes
from sys import getsizeof

class DWDataReader:
    def __init__(self, dll_path='.\\DWDataReaderLib64.dll', file_paths = "C:\\Users\\jvv20\\Vibra\\DeweSoftData\\Data\\"):
        self.lib = cdll.LoadLibrary(dll_path)
        self.file_paths = file_paths
        if self.lib.DWInit() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWInit() failed")
        self.add_reader()
        self.ensure_everything_is_ok()

    def __del__(self):
        self.close()


    def raise_error(self, message):
        raise Exception(f"DWDataReader: {message}")

    def add_reader(self):
        print("reader added")
        if self.lib.DWAddReader() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWAddReader() failed")

    def open_data_file(self, file_name):
        print("file data opend")
        full_name = self.file_paths + file_name + ".dxd"
        file_name_c = c_char_p(full_name.encode())
        self.file_info = DWFileInfo(0, 0, 0)
        if self.lib.DWOpenDataFile(file_name_c, c_void_p(addressof(self.file_info))) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWOpenDataFile() failed")

    def get_measurement_info(self) -> DWMeasurementInfo: 
        measurement_info = DWMeasurementInfo(0, 0, 0, 0)
        if self.lib.DWGetMeasurementInfo(c_void_p(addressof(measurement_info))) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWGetMeasurementInfo() failed")

        return measurement_info

    def get_channel_list(self) -> tuple[list, int]:
        print("called get channel list")
        num_channels = self.lib.DWGetChannelListCount()
        if num_channels == -1:
            self.raise_error("DWGetChannelListCount() failed")
        
        ch_list = (DWChannel * num_channels)()
        if self.lib.DWGetChannelList(byref(ch_list)) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWGetChannelList() failed")

        return ch_list, num_channels

    def get_scaled_samples(self, channel_index):
        dw_ch_index = c_int(channel_index)
        sample_cnt = c_int()
        sample_cnt = self.lib.DWGetScaledSamplesCount(dw_ch_index)
        if sample_cnt < 0:
            self.raise_error("DWGetScaledSamplesCount() failed")
        data = create_string_buffer(DOUBLE_SIZE * sample_cnt * self.channel_list[channel_index].array_size)
        time_stamp = create_string_buffer(DOUBLE_SIZE * sample_cnt)
        print("Executed time_stamp")
        p_data = cast(data, POINTER(c_double))
        print("Executed p_data")
        p_time_stamp = cast(time_stamp, POINTER(c_double))
        print("Executed p_time_stamp")
        if self.lib.DWGetScaledSamples(dw_ch_index, c_int64(0), sample_cnt, p_data, p_time_stamp) != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWGetScaledSamples() failed")
        return [(p_time_stamp[i], p_data[i]) for i in range(sample_cnt)]
    
    def ensure_everything_is_ok(self):
        if self.lib.DWGetNumReaders(byref(c_int())) != DWStatus.DWSTAT_OK.value:
            self.raise_error("Data readers are not initialized correctly")
        print("All systems are operational.")
    
    def get_measurements_as_dataframe(self) -> pd.DataFrame:
        print("get measurements as df called")
        ch_list, num_channels = self.get_channel_list()
        return_data = {}
        #data["Time"] = np.array([val[0] for val in self.get_scaled_samples(0)])
        for i in range(0, num_channels):
            # basic channel properties
            #print("************************************************")
            #print("Channel #%d" % i)
            #print("************************************************")
            #print("Index: %d" % ch_list[i].index)
            #print("Name: %s" % ch_list[i].name.decode())
            #print("Unit: %s" % ch_list[i].unit.decode())
            #print("Description: %s" % ch_list[i].description.decode())

            # channel factors
            idx = c_int(ch_list[i].index)
            ch_scale = c_double()
            ch_offset = c_double()
            if self.lib.DWGetChannelFactors(idx, byref(ch_scale), byref(ch_offset)) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetChannelFactors() failed")

            #print("Scale: %.2f" % ch_scale.value)
            #print("Offset: %.2f" % ch_offset.value)

            # channel type
            max_len = c_int(INT_SIZE)
            buff = create_string_buffer(max_len.value)
            p_buff = cast(buff, POINTER(c_void_p))
            if self.lib.DWGetChannelProps(idx, c_int(DWChannelProps.DW_CH_TYPE.value), p_buff, byref(max_len)) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetChannelProps() failed")
            ch_type = cast(p_buff, POINTER(c_int)).contents

            if ch_type.value == DWChannelType.DW_CH_TYPE_SYNC.value:
                print("Channel type: sync")
            elif ch_type.value == DWChannelType.DW_CH_TYPE_ASYNC.value:
                print("Channel type: async")
            elif ch_type.value == DWChannelType.DW_CH_TYPE_SV.value:
                print("Channel type: single value")
            else:
                print("Channel type: unknown")

            # channel data type
            if self.lib.DWGetChannelProps(idx, c_int(DWChannelProps.DW_DATA_TYPE.value), p_buff, byref(max_len)) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetChannelProps() failed")
            data_type = cast(p_buff, POINTER(c_int)).contents
            #print("Data type: %s" % DWDataType(data_type.value).name)

            # channel long name length
            channel_long_name_len_buff = create_string_buffer(INT_SIZE)
            if self.lib.DWGetChannelProps(idx, c_int(DWChannelProps.DW_CH_LONGNAME_LEN.value), channel_long_name_len_buff, byref(max_len)) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetChannelProps() failed")
            long_name_len = cast(channel_long_name_len_buff, POINTER(c_int)).contents

            # channel long name
            channel_long_name_buff = create_string_buffer(long_name_len.value)
            if self.lib.DWGetChannelProps(idx, c_int(DWChannelProps.DW_CH_LONGNAME.value), channel_long_name_buff, byref(long_name_len)) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetChannelProps() failed")
            #print("Long name : %s" % channel_long_name_buff.value.decode())

            # channel xml length
            channel_xml_len_buff = create_string_buffer(INT_SIZE)
            if self.lib.DWGetChannelProps(idx, c_int(DWChannelProps.DW_CH_XML_LEN.value), channel_xml_len_buff, byref(max_len)) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetChannelProps() failed")
            xml_len = cast(channel_xml_len_buff, POINTER(c_int)).contents

            # channel xml
            channel_xml_buff = create_string_buffer(xml_len.value)
            if self.lib.DWGetChannelProps(idx, c_int(DWChannelProps.DW_CH_XML.value), channel_xml_buff, byref(xml_len)) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetChannelProps() failed")
            #print("Xml : %s" % channel_xml_buff.value.decode())

            # number of samples
            dw_ch_index = c_int(ch_list[i].index)
            sample_cnt = c_int()
            sample_cnt = self.lib.DWGetScaledSamplesCount(dw_ch_index)
            if sample_cnt < 0:
                DWRaiseError("DWDataReader: DWGetScaledSamplesCount() failed")
            #print("Num. samples: %d" % sample_cnt)

            # get actual data
            data = create_string_buffer(
                DOUBLE_SIZE * sample_cnt * ch_list[i].array_size)
            time_stamp = create_string_buffer(DOUBLE_SIZE * sample_cnt)
            p_data = cast(data, POINTER(c_double))
            p_time_stamp = cast(time_stamp, POINTER(c_double))
            if self.lib.DWGetScaledSamples(dw_ch_index, c_int64(0), sample_cnt, p_data, p_time_stamp) != DWStatus.DWSTAT_OK.value:
                DWRaiseError("DWDataReader: DWGetScaledSamples() failed")


            if "Time" not in return_data.keys():
                return_data["Time"] = np.array([p_time_stamp[j] for j in range(sample_cnt)])
                
            # diplay data
            # Create a dictionary to hold data temporarily
            channel_data = {}

            for j in range(sample_cnt):
                name = ch_list[i].name.decode()  # Extract channel name
                values = [p_data[j * ch_list[i].array_size + k] for k in range(ch_list[i].array_size)]  # Extract values

                if name not in channel_data:
                    channel_data[name] = []  # Initialize list if not present
                
                channel_data[name].append(values)  # Store raw values

            # Convert lists to NumPy arrays
            for name, values in channel_data.items():
                return_data[name] = np.array(values)

        #print("after for loop")
        #print(len(data))
        return return_data
    
    def close(self):
        print("close called")
        if self.lib.DWCloseDataFile() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWCloseDataFile() failed")
        if self.lib.DWDeInit() != DWStatus.DWSTAT_OK.value:
            self.raise_error("DWDeInit() failed")
        _ctypes.FreeLibrary(self.lib._handle)
        del self.lib
