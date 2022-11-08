from processors import sort_traceset
from processors import ID_wise_mean
from sstats import SSOD
from sfileengine import H5FileEngine
import matplotlib.pyplot as plt

path = r'..\..\sscametric-strainfunctions\ASCAD_dataset\ASCAD.h5'
fileEngine = H5FileEngine(path, group='/Profiling_traces', trace_processor='abs')

print (fileEngine[0][1])
plt.plot(fileEngine[0][0])
plt.plot(fileEngine[1][0])
plt.plot(fileEngine[2][0])
plt.plot(fileEngine[3][0])
plt.plot(fileEngine[4][0])
plt.show()

result = sort_traceset(fileEngine)

print (result.shape)
print (result[0].shape)
print (result[1].shape)

mean_total, op_mean, op_var = ID_wise_mean(fileEngine)
print (mean_total)

ssod = SSOD(op_mean)

plt.plot(ssod)
plt.show()

fileEngine.close()