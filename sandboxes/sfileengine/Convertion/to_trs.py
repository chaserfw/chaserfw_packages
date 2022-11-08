from sfileengine import H5FileEngine
from sfileengine.processors import AbsTrimming
trimming = AbsTrimming(10, 15)

dataset_path = r'..\..\math\processor\ASCAD_var_desync100-whole-clean-scaled.h5'
file = H5FileEngine(dataset_path, group='Profiling_traces', trace_processor=trimming)

file.to_trsfile()
file.close()