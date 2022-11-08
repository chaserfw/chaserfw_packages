from sfileengine import TRSFileEngine
from sfileengine import H5FileEngine
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from sstats import snr_byte
import tables



#path = r'D:\ASCAD_EPF_20201222-105722.trs'
#path = r'C:\Users\slpaguada\Datasets\ASCAD\ATMEGA_AES_v1\ATM_AES_v1_variable_key\ASCAD_data\ASCAD_databases\ASCAD_var.h5'
#path = r'D:\SCA\datasets\ASCAD_var_desync100.h5'
path = r'D:\ASCAD_APF_20201129-151305.h5'

tfile = tables.open_file(path)
print(tfile.root.Profiling_traces.traces.shape)
print (help(tfile.root.Profiling_traces.traces))
tfile.close()

#file = TRSFileEngine(path)
file = H5FileEngine(path, group="/Profiling_traces", label_iteration=True)

print (file.TotalTraces)

traces = file[:4]
#print (traces)
#print (traces[:][1])
traces = np.array(traces)
#print (traces[:,0][:3])
print (traces[:,0].shape)
print (traces[:,1])
print (traces[:,1].shape)


print (file.TotalSamples)
print (file.OriginalPath)
print (file.FileName)
print (file.Path)

print (file.FileType)
#print (file[1])
print (len(file))

print (file.Group)

for i in trange(10):
	print (file[i][0][0])



file.close()