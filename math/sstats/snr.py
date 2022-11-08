from sutils import trange
from scryptoutils import AES_Sbox
##################################################################################
"""
def snr_byte(trs_file, byte_pos_list):
	from sklearn.preprocessing import StandardScaler
	import numpy as np

	# Get number of traces
	n_traces = len(trs_file)

	sc = StandardScaler()
	all_byte_slots = {}
	byte_counter   = {}
	matrix_result = []
	for byte_pos in byte_pos_list:
		# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
		for i in trange(n_traces, desc='[INFO *SNRByte*]: Computing SNR for {} byte position'.format(byte_pos)):
			# First time the byte appears in the dictionary
			selected_byte = np.frombuffer(trs_file[i].data, dtype='uint8')[byte_pos]
			if not (selected_byte in all_byte_slots):
				all_byte_slots[selected_byte] = StandardScaler()
				byte_counter[selected_byte] = 0
			
			# partially fit the scaler with the trace grouped by the key, increase by one the trace counter
			all_byte_slots[selected_byte].partial_fit(np.array(trs_file[i]).reshape(1, -1))
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
		matrix_result.append([np.true_divide(var_mean, mean_var), var_mean, mean_var])

	return matrix_result

def snr_masked_sbox(trs_file, plaintext_pos, key_pos):
	from sklearn.preprocessing import StandardScaler
	import numpy as np

	# Get number of traces
	n_traces = len(trs_file)

	sc = StandardScaler()
	all_byte_slots = {}
	byte_counter   = {}

	# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
	for i in trange(n_traces, desc='[INFO *SNRMaskedByte*]: Computing partial fit'):
		# Metadata vector
		metadata_vector = np.frombuffer(trs_file[i].data, dtype='uint8')

		# First time the byte appears in the dictionary
		selected_byte = AES_Sbox[np.frombuffer(trs_file[i].data, dtype='uint8')[plaintext_pos] ^ np.frombuffer(trs_file[i].data, dtype='uint8')[key_pos]]
		if not (selected_byte in all_byte_slots):
			all_byte_slots[selected_byte] = StandardScaler()
			byte_counter[selected_byte] = 0
		
		# partially fit the scaler with the trace grouped by the key, increase by one the trace counter
		all_byte_slots[selected_byte].partial_fit(np.array(trs_file[i]).reshape(1, -1))
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

def snr_unmasked_sbox(trs_file, plaintext_pos, key_pos, mask_pos):
	from sklearn.preprocessing import StandardScaler
	import numpy as np

	# Get number of traces
	n_traces = len(trs_file)

	sc = StandardScaler()
	all_byte_slots = {}
	byte_counter   = {}

	# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
	for i in trange(n_traces, desc='[INFO *SNRUnMaskedByte*]: Computing partial fit'):
		# Metadata vector
		metadata_vector = np.frombuffer(trs_file[i].data, dtype='uint8')

		# First time the byte appears in the dictionary
		selected_byte = AES_Sbox[metadata_vector[plaintext_pos] ^ metadata_vector[key_pos]] ^ metadata_vector[mask_pos]
		if not (selected_byte in all_byte_slots):
			all_byte_slots[selected_byte] = StandardScaler()
			byte_counter[selected_byte] = 0
		
		# partially fit the scaler with the trace grouped by the key, increase by one the trace counter
		all_byte_slots[selected_byte].partial_fit(np.array(trs_file[i]).reshape(1, -1))
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

"""
##################################################################################
def snr_var_mean(file_engine, ntraces=None, ftrace=None, leavepb=True):
	from sklearn.preprocessing import StandardScaler
	import numpy as np

	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces
	ftrace = ftrace if ftrace is not None else 0

	sc = StandardScaler()
	for index in trange(n_traces, desc='[INFO *SNRMeanVar*]', leave=leavepb):
		sc.partial_fit(np.array(np.abs(file_engine[ftrace+index][0])).reshape(1, -1))
	snr  = sc.mean_/np.sqrt(sc.var_)
	snr[np.isnan(snr)] = 0
	return snr
##################################################################################
def snr_byte(file_engine, byte_pos_list, ntraces=None):
	from sklearn.preprocessing import StandardScaler
	import numpy as np

	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces

	sc = StandardScaler()
	all_byte_slots = {}
	byte_counter   = {}
	matrix_result = []
	for byte_pos in byte_pos_list:
		# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
		for i in trange(n_traces, desc='[INFO *SNRByte*]: Computing SNR for {} byte position'.format(byte_pos)):
			# First time the byte appears in the dictionary
			selected_byte = file_engine[i][1][byte_pos]
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
		matrix_result.append([np.true_divide(var_mean, mean_var), var_mean, mean_var])

	return matrix_result

##################################################################################
def snr_masked_sbox(file_engine, plaintext_pos, key_pos, ntraces=None):
	'''Compute the SNR Sbox variance, the function assumes that it is a masked Sbox and
	it does not require the mask position. This allows the function to be used in
	scenarios where the Sbox is masked or not.
	'''
	from sklearn.preprocessing import StandardScaler
	import numpy as np

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

##################################################################################
def snr_unmasked_sbox(file_engine, plaintext_pos, key_pos, mask_pos, ntraces=None):
	from sklearn.preprocessing import StandardScaler
	import numpy as np

	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces

	sc = StandardScaler()
	all_byte_slots = {}
	byte_counter   = {}

	# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
	for i in trange(n_traces, desc='[INFO *SNRUnMaskedByte*]: Computing partial fit'):
		# Metadata vector
		metadata_vector = file_engine[i][1]

		# First time the byte appears in the dictionary
		selected_byte = AES_Sbox[metadata_vector[plaintext_pos] ^ metadata_vector[key_pos]] ^ metadata_vector[mask_pos]
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