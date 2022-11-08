
from processors import ID_wise_mean_balanced
from sfileengine import H5FileEngine

path = r'..\..\sscametric-strainfunctions\ASCAD_dataset\ASCAD.h5'
databalancer_file = r'..\..\sscametric-strainfunctions\ASCAD_dataset\index_map_ASCAD.pickle'

fileEngine = H5FileEngine(path, group='/Profiling_traces')

mean_total, all_means, _, _ = ID_wise_mean_balanced(fileEngine, databalancer_file, n_indexes=10)
print (mean_total.shape)
print (all_means.shape)

fileEngine.close()