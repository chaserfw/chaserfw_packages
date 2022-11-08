from sstats import SSOD_from_fileengine
from sfileengine import H5FileEngine
import matplotlib.pyplot as plt

dataset_path = r'..\template\ASCAD-compress.h5'
fileEngine = H5FileEngine(dataset_path, group='/Profiling_traces')

ssod = SSOD_from_fileengine(fileEngine, clustered_traces=True)
fileEngine.close()

plt.plot(ssod)
plt.show()