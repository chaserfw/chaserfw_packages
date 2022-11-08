import numpy as np
from sutils import trange
from sfileengine import FileEngine
from processors import ID_wise_mean
	
def SSOD(clustered_means:np.ndarray) -> np.ndarray:
	"""Computes the sums of absolute differences from a numpy array comprises 
	a clustered means. It is normally applied after computing an op_wise_mean, 
	so for AES SCA attack models clustered_means parameter is a matrix of
	256xlenght_of_trace (or 9xlenght_of_trace in HW case). 
	It was not proved in attacks base on other crypto algorithms

	:param clustered_means: Clustered means numpy array
	:type clustered_means: :class:`np.ndarray`
	
	:return: the sums of absolute differences
	:rtype: :class:`np.ndarray`
	"""
	shape = clustered_means.shape
	__temp_sum_diff = np.zeros(shape[1])

	for i in trange(shape[0], desc='[INFO *SSOD*]:Computing sums of diff'):
		for j in range(i):
			__temp_sum_diff += np.abs(clustered_means[i] - clustered_means[j])
	
	return __temp_sum_diff


def SSOD_from_fileengine(dataset:FileEngine, plaintext_pos:int=None, key_pos:int=None, 
						 ntraces:int=None, label:bool=True, 
						 clustered_traces:bool=False) -> np.ndarray:

	clustered_means = ID_wise_mean(dataset, plaintext_pos, key_pos, 
								   ntraces, label, clustered_traces)
	
	return SSOD(clustered_means[1])