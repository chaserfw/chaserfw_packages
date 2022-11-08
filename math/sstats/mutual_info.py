"""
@author: Servio Paguada
@email: serviopaguada@gmail.com
"""
from tqdm import trange
import numpy as np

def shannon_entropy(rv):
	rv_normalized = np.true_divide(rv, float(np.sum(rv)))
	rv_normalized = rv_normalized[np.nonzero(rv_normalized)]
	H = -np.sum(rv_normalized * np.log2(rv_normalized))
	return H

def vector_shannon_entropy(rv):
	pass

def histo_based_entropy(X, bins):
	c_X  = np.histogram(X, bins)[0]
	return shannon_entropy(c_X)

def calc_MI(X, Y, bins):
	c_XY = np.histogram2d(X, Y, bins)[0]
	c_X  = np.histogram(X, bins)[0]
	c_Y  = np.histogram(Y, bins)[0]

	H_X = shannon_entropy(c_X)
	H_Y = shannon_entropy(c_Y)
	H_XY = shannon_entropy(c_XY)

	MI = H_X + H_Y - H_XY
	return MI

def sk_calc_MI(X, Y, bins):
	from sklearn.metrics import mutual_info_score
	c_xy = np.histogram2d(X, Y, bins)[0]
	mi = mutual_info_score(None, None, contingency=c_xy)
	mi = mi/np.log(2)
	return mi

def sc_calc_MI(X, Y):
	from sklearn.preprocessing import StandardScaler

	all_byte_slots = {}

	for i in trange(X.shape[0], desc='[INFO *MI*]: Computing partial fit'):
		# First time the byte appears in the dictionary
		selected_byte = AES_Sbox[metadata_vector[plaintext_pos] ^ metadata_vector[key_pos]]
		if not (selected_byte in all_byte_slots):
			all_byte_slots[selected_byte] = StandardScaler()
			byte_counter[selected_byte] = 0

def snr_masked_sbox(file_engine, plaintext_pos, key_pos, ntraces=None):
	from sklearn.preprocessing import StandardScaler	

	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces

	sc = StandardScaler()
	all_byte_slots = {}
	byte_counter   = {}

	# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
	for i in trange(n_traces, desc='[INFO *SNRMaskedByte*]: Computing partial fit'):
		# Metadata vector
		metadata_vector = file_engine[i][1]

		# First time the byte appears in the dictionary
		selected_byte = AES_Sbox[metadata_vector[plaintext_pos] ^ metadata_vector[key_pos]]
		if not (selected_byte in all_byte_slots):
			all_byte_slots[selected_byte] = StandardScaler()
			byte_counter[selected_byte] = 0
		
		# partially fit the scaler with the trace grouped by the key, increase by one the trace counter
		all_byte_slots[selected_byte].partial_fit(np.array(file_engine[i][0]).reshape(1, -1))
		byte_counter[selected_byte] += 1
	
	# Matrix of all means and all vars, it's a list, which means that allows us to use with any crypto algorithm and not only 256 key byte algorithm
	all_means = []
	all_vars = []
	var_mean = None
	mean_var = None
	for key, scalers in all_byte_slots.items():
		all_means.append(scalers.mean_)
		all_vars.append(scalers.var_)

	var_mean = np.var(np.array(all_means, dtype=np.float), 0)
	mean_var = np.mean(np.array(all_vars, dtype=np.float), 0)

	return (np.true_divide(var_mean, mean_var), var_mean, mean_var)