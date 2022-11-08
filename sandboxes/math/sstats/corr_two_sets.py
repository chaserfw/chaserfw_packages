from sstats.file_engine_process import compute_corr_two_sets
from sstats.file_engine_process import compute_convcorr_two_sets
import math
from sfileengine import MemoryFileEngine
import numpy as np
import matplotlib.pyplot as plt


coeffs = np.random.rand(100, 50)
signal1 = []
for coeff in coeffs:
    signal1.append([math.sin(x) for x in math.pi*coeff])
signalFE1 = MemoryFileEngine(np.array(signal1))

coeffs = np.random.rand(100, 50)
signal2 = []
for coeff in coeffs:
    signal2.append([math.sin(x) for x in math.pi*coeff])
signalFE2 = MemoryFileEngine(np.array(signal2))


#corr = compute_convcorr_two_sets(signalFE1, signalFE2)
#plt.plot(corr)
#plt.show()

corr = compute_corr_two_sets(signalFE1, signalFE2)
plt.plot(corr)
plt.show()
