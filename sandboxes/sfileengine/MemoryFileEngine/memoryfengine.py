from sklearn.preprocessing import StandardScaler
from sfileengine import MemoryFileEngine
import numpy as np
from sstats import snr_byte
from sstats import snr_masked_sbox
from tqdm import trange
from sstats import calc_MI
from sstats import sk_calc_MI
from sstats import H
from sstats import entropy

traces = np.random.rand(1000, 700)
metadata = np.random.randint(0, 256, size=(1000, 34))
memFileEngine = MemoryFileEngine(traces, metadata)

mu    = 0
sigma = 0.1
X     = np.random.randint(0, 256, size=(1000))#np.random.normal(mu, sigma, size=(1000))
Y     = np.random.randint(0, 256, size=(1000))

print ('max_value', np.max(metadata))

sc = StandardScaler()
sc2 = StandardScaler()
for i in trange(X.shape[0]):
    sc.partial_fit(X[i].reshape(-1, 1))
    sc2.partial_fit(Y[i].reshape(-1, 1))

print (np.cov(X, Y))
print (sc.var_)
print (sc2.var_)

suma = np.zeros(shape=(X.shape[0]), dtype=np.float64)
sc3 = StandardScaler()
for i in trange(X.shape[0]):
    suma[i] = (X[i] - sc.mean_) * (Y[i] - sc2.mean_)
    sc3.partial_fit(suma[i].reshape(-1, 1))

print (np.average(suma))
print (sc3.mean_)
print ('true_divide:', np.true_divide(np.sum(suma), X.shape[0] - 1))
print ('true_divide:', np.cov(X, Y)[0, 1])

#for i in trange():

print (calc_MI(X, Y, 256))
print (sk_calc_MI(X, Y, 256))

x = [[1.3],[3.7],[5.1],[2.4]]
print (H(x, 5))
print (entropy(x, k=3))
#x = [[1.3],[3.7],[5.1],[2.4]]
#print (entropy(x, base=2))



"""
El rollo es penoso, estoy implementandola usando vectorización por el calculo con demasiados datos y la memoria no soporta tantos datos y tal, pasa que mi resultado no es igual al de las funciones ya implementadas.

Te muestro, esta es mi implementación:
product[i] = (X[i] - mean_X) * (Y[i] - mean_Y), para i = 1,2,...,n
luego prodcut
"""

