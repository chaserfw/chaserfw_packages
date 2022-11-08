from sfileengine import MemoryFileEngine
import numpy as np

traces = np.random.rand(1000, 700)
metadata = np.random.randint(0, 256, size=(1000, 34))
memFileEngine = MemoryFileEngine(traces, metadata)

trace_meta = np.array(memFileEngine[0:3], dtype=np.ndarray)
print (trace_meta)
print ('----------')
print (trace_meta[:,0])

