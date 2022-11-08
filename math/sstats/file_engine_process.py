import numpy as np
from sutils.stqdm import trange
from sklearn.preprocessing import StandardScaler

##################################################################################
##################################################################################
# FILE ENGINE SUPPORT
##################################################################################
##################################################################################

##################################################################################
# Byte position correlation
##################################################################################
def compute_mean_std(file_engine, byte_pos_list, ntraces=None):
	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces

	trace_mean_std = StandardScaler()
	trace_mean_std_ready = False
	all_byte_slots = {}
	for byte_pos in byte_pos_list:
		# Create a StandardScaler for each byte position
		byte_pos_scaler = StandardScaler()
	
		for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: Computing mean and std of {} byte position'.format(byte_pos)):			
			# partially fit the scaler of a byte position
			byte_pos_scaler.partial_fit(np.array(file_engine[i][1][byte_pos]).reshape(1, -1))

			# Check if the trace standard scaler was not fit yet
			if not trace_mean_std_ready:
				trace_mean_std.partial_fit(np.array(file_engine[i][0]).reshape(1, -1))
			
		trace_mean_std_ready = True

		# Store each mean and std
		all_byte_slots[byte_pos] = []
		all_byte_slots[byte_pos].append(byte_pos_scaler.mean_)
		all_byte_slots[byte_pos].append(byte_pos_scaler.var_)

	return ([trace_mean_std.mean_, trace_mean_std.var_], all_byte_slots)

##################################################################################
def compute_corr_byte_pos(file_engine, n_traces, metadata_byte, samples_mean, samples_std, metadata_mean, metadata_std):
	# Get group
	n_samples = file_engine.TotalSamples
	metadata_dim = 0    
	samples_corr    = np.zeros(shape=(n_samples,),    dtype=np.float64)
	single_metadata = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	
	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: Computing corr - byte: {}'.format(metadata_byte)):
		meta         = file_engine[i][1]
		samples_corr = np.add(samples_corr, (file_engine[i][0] - samples_mean) * (meta[metadata_byte] - metadata_mean))
	if metadata_std == 0:
		print ('[WARNING *FileEngineCorrProcess*]: Metadata standard deviation of byte {} is zero'.format(metadata_byte))
		print ('[INFO *FileEngineCorrProcess*]: Returning zero correlation')
	else:
		samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * metadata_std))
	return samples_corr
##################################################################################
def compute_corr(file_engine, vb_list=[], n_traces=None):
	
	all_mean_std    = compute_mean_std(file_engine, vb_list, n_traces)
	samples_mean    = all_mean_std[0][0]
	samples_std     = np.sqrt(all_mean_std[0][1])
	all_bytes_slots = all_mean_std[1]

	samples_corr = []
	for i, (vb, vb_mean_std_list) in enumerate(all_bytes_slots.items()):
		metadata_mean = vb_mean_std_list[0]
		metadata_std  = np.sqrt(vb_mean_std_list[1])
		samples_corr.append(compute_corr_byte_pos(file_engine, n_traces, vb, samples_mean, samples_std, metadata_mean, metadata_std))
		
	return samples_corr

##################################################################################
# AES Sbox
##################################################################################
def compute_mean_std_sbox(file_engine, plaintext_byte, key_byte, ntraces=None):
	from scryptoutils import AES_Sbox
	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces	
	
	# Create a StandardScaler for each byte position
	trace_mean_std  = StandardScaler()
	aes_sbox_scaler = StandardScaler()

	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: mean and std sbox(pt:{},key:{})'.format(plaintext_byte, key_byte)):
		# partially fit the scaler of a byte position
		meta = file_engine[i][1]
		aes_sbox_scaler.partial_fit(np.array(AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]]).reshape(1, -1))
		trace_mean_std.partial_fit(np.array(file_engine[i][0]).reshape(1, -1))

	return ([trace_mean_std.mean_, trace_mean_std.var_], [aes_sbox_scaler.mean_, aes_sbox_scaler.var_])

def compute_corr_sbox(file_engine, plaintext_byte, key_byte, n_traces=None):
	"""Correlation based on AES Sbox
	"""
	from scryptoutils import AES_Sbox

	# Get group
	n_samples = file_engine.TotalSamples
	samples_corr    = np.zeros(shape=(n_samples,), dtype=np.float64)

	all_means_std = compute_mean_std_sbox(file_engine, plaintext_byte, key_byte, n_traces)
	samples_mean  = all_means_std[0][0]
	samples_std   = np.sqrt(all_means_std[0][1])
	metadata_mean = all_means_std[1][0]
	metadata_std  = np.sqrt(all_means_std[1][1])
	
	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: computing corr sbox(pt:{},key:{})'.format(plaintext_byte, key_byte)):
		meta         = file_engine[i][1]
		samples_corr = np.add(samples_corr, (file_engine[i][0] - samples_mean) * (AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] - metadata_mean))
		
	if np.count_nonzero(metadata_std) == 0 or np.count_nonzero(samples_std) == 0:
		print ('[WARNING *FileEngineCorrProcess*]: Metadata or samples standard deviation of AES Sbox plaintext {} and key {} is zero'.format(plaintext_byte, key_byte))
		print ('[INFO *FileEngineCorrProcess*]: Returning zero correlation')
	else:
		samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * metadata_std))
	return samples_corr


##################################################################################
# AES UnSbox
##################################################################################
def compute_mean_std_unsbox(file_engine, plaintext_byte, key_byte, mask_byte, ntraces=None):
	from scryptoutils import AES_Sbox
	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces	
	
	# Create a StandardScaler for each byte position
	trace_mean_std  = StandardScaler()
	aes_sbox_scaler = StandardScaler()

	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: mean and std sbox(pt:{}^key:{})^m:{}'.format(plaintext_byte, key_byte, mask_byte)):
		# partially fit the scaler of a byte position
		meta = file_engine[i][1]
		aes_sbox_scaler.partial_fit(np.array(AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] ^ meta[mask_byte]).reshape(1, -1))
		trace_mean_std.partial_fit(np.array(file_engine[i][0]).reshape(1, -1))

	return ([trace_mean_std.mean_, trace_mean_std.var_], [aes_sbox_scaler.mean_, aes_sbox_scaler.var_])

def compute_corr_unsbox(file_engine, plaintext_byte, key_byte, mask_byte, n_traces=None):
	"""Correlation based on AES Sbox
	"""
	from scryptoutils import AES_Sbox

	# Get group
	n_samples = file_engine.TotalSamples
	samples_corr    = np.zeros(shape=(n_samples,), dtype=np.float64)

	all_means_std = compute_mean_std_unsbox(file_engine, plaintext_byte, key_byte, mask_byte, n_traces)
	samples_mean  = all_means_std[0][0]
	samples_std   = np.sqrt(all_means_std[0][1])
	metadata_mean = all_means_std[1][0]
	metadata_std  = np.sqrt(all_means_std[1][1])
	
	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: computing corr sbox(pt:{}^key:{})^m:{}'.format(plaintext_byte, key_byte, mask_byte)):
		meta         = file_engine[i][1]
		samples_corr = np.add(samples_corr, (file_engine[i][0] - samples_mean) * ((AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]]  ^ meta[mask_byte])- metadata_mean))
		
	if np.count_nonzero(metadata_std) == 0 or np.count_nonzero(samples_std) == 0:
		print ('[WARNING *FileEngineCorrProcess*]: Metadata or samples standard deviation of AES Sbox pt {}, key {}, and mask {} is zero'.format(plaintext_byte, key_byte, mask_byte))
		print ('[INFO *FileEngineCorrProcess*]: Returning zero correlation')
	else:
		samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * metadata_std))
	return samples_corr

##################################################################################
# AES double UnSbox
##################################################################################
def compute_mean_std_double_unsbox(file_engine, plaintext_byte, key_byte, mask_byte_1, mask_byte_2, ntraces=None):
	from scryptoutils import AES_Sbox
	# Get number of traces
	n_traces = len(file_engine) if ntraces is None else ntraces	
	
	# Create a StandardScaler for each byte position
	trace_mean_std  = StandardScaler()
	aes_sbox_scaler = StandardScaler()

	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: mean and std sbox(pt:{}^key:{}^m1:{})^m2:{}'.format(plaintext_byte, key_byte, mask_byte_1, mask_byte_2)):
		# partially fit the scaler of a byte position
		meta = file_engine[i][1]
		aes_sbox_scaler.partial_fit(np.array(AES_Sbox[meta[plaintext_byte] ^ meta[key_byte] ^ meta[mask_byte_1]] ^ meta[mask_byte_2]).reshape(1, -1))
		trace_mean_std.partial_fit(np.array(file_engine[i][0]).reshape(1, -1))

	return ([trace_mean_std.mean_, trace_mean_std.var_], [aes_sbox_scaler.mean_, aes_sbox_scaler.var_])

def compute_corr_double_unsbox(file_engine, plaintext_byte, key_byte, mask_byte_1, mask_byte_2, n_traces=None):
	"""Correlation based on AES Sbox
	"""
	from scryptoutils import AES_Sbox

	# Get group
	n_samples = file_engine.TotalSamples
	samples_corr    = np.zeros(shape=(n_samples,), dtype=np.float64)

	all_means_std = compute_mean_std_double_unsbox(file_engine, plaintext_byte, key_byte, mask_byte_1, mask_byte_2, n_traces)
	samples_mean  = all_means_std[0][0]
	samples_std   = np.sqrt(all_means_std[0][1])
	metadata_mean = all_means_std[1][0]
	metadata_std  = np.sqrt(all_means_std[1][1])
	
	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: computing corr sbox(pt:{}^key:{}^m1:{})^m2:{}'.format(plaintext_byte, key_byte, mask_byte_1, mask_byte_2)):
		meta         = file_engine[i][1]
		samples_corr = np.add(samples_corr, (file_engine[i][0] - samples_mean) * ((AES_Sbox[meta[plaintext_byte] ^ meta[key_byte] ^ meta[mask_byte_1]] ^ meta[mask_byte_2]) - metadata_mean))
		
	if np.count_nonzero(metadata_std) == 0 or np.count_nonzero(samples_std) == 0:
		print ('[WARNING *FileEngineCorrProcess*]: Metadata or samples standard deviation of AES Sbox pt:{}, key:{}, m1:{}, and m2:{} is zero'.format(plaintext_byte, key_byte, mask_byte_1, mask_byte_2))
		print ('[INFO *FileEngineCorrProcess*]: Returning zero correlation')
	else:
		samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * metadata_std))
	return samples_corr

#---------------------------------------------------------------------------------
# Correlate two sets
#---------------------------------------------------------------------------------
def _compute_mean_std_sets(file_engine, file_engine_2, ntraces):
	fe_mean_std  = StandardScaler()
	fe_mean_std_2 = StandardScaler()
		
	for i in trange(ntraces, desc='[INFO *FileEngineCorrProcess*]: Computing mean and std of two sets'):
		# partially fit the scaler of a byte position
		fe_mean_std.partial_fit(np.array(file_engine[i][0]).reshape(1, -1))
		fe_mean_std_2.partial_fit(np.array(file_engine_2[i][0]).reshape(1, -1))

	return ([fe_mean_std.mean_, fe_mean_std.var_], [fe_mean_std_2.mean_, fe_mean_std_2.var_])

def compute_corr_two_sets(file_engine, file_engine_2, n_traces=None, mode='same'):
	"""Correlation between two signals, signal must have the same lenght. 
	For an implementation that uses signals with different lenghts, refer to 
	compute_convcorr_two_sets function
	"""
	# Get number of traces
	n_traces = len(file_engine) if n_traces is None else n_traces

	# Get group
	n_samples = file_engine.TotalSamples if file_engine.TotalSamples > file_engine_2.TotalSamples else file_engine_2.TotalSamples
	samples_corr    = np.zeros(shape=(n_samples,), dtype=np.float64)

	all_means_std   = _compute_mean_std_sets(file_engine, file_engine_2, n_traces)
	samples_std     = np.sqrt(all_means_std[0][1])
	samples_mean    = all_means_std[0][0]
	samples_std_2   = np.sqrt(all_means_std[1][0])
	samples_mean_2  = all_means_std[1][1]
	
	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: computing two sets corr'):
		samples_corr = np.add(samples_corr, (file_engine[i][0] - samples_mean) * (file_engine_2[i][0] - samples_mean_2))
		
	if np.count_nonzero(samples_std_2) == 0 or np.count_nonzero(samples_std) == 0:
		print ('[WARNING *FileEngineCorrProcess*]: One of the two samples standard deviation is zero')
		print ('[INFO *FileEngineCorrProcess*]: Returning zero correlation')
	else:
		samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * samples_std_2))
	return samples_corr

def compute_convcorr_two_sets(file_engine, file_engine_2, n_traces=None, mode='same'):
	"""Correlation between two signals, it uses convolutions to multiply the signals
	"""
	# Get number of traces
	n_traces = len(file_engine) if n_traces is None else n_traces

	# Get group
	n_samples = file_engine.TotalSamples if file_engine.TotalSamples > file_engine_2.TotalSamples else file_engine_2.TotalSamples
	samples_corr    = np.zeros(shape=(n_samples,), dtype=np.float64)

	all_means_std   = _compute_mean_std_sets(file_engine, file_engine_2, n_traces)
	samples_std     = np.sqrt(all_means_std[0][1])
	samples_mean    = all_means_std[0][0]
	samples_std_2   = np.sqrt(all_means_std[1][0])
	samples_mean_2  = all_means_std[1][1]
	
	for i in trange(n_traces, desc='[INFO *FileEngineCorrProcess*]: computing two sets corr'):
		#samples_corr = np.add(samples_corr, (file_engine[i][0] - samples_mean) * (file_engine_2[i][0] - samples_mean_2))
		# Computing using convolutions
		samples_corr = np.add(samples_corr, np.convolve((file_engine[i][0] - samples_mean), (file_engine_2[i][0] - samples_mean_2), mode))
		
	if np.count_nonzero(samples_std_2) == 0 or np.count_nonzero(samples_std) == 0:
		print ('[WARNING *FileEngineCorrProcess*]: One of the two samples standard deviation is zero')
		print ('[INFO *FileEngineCorrProcess*]: Returning zero correlation')
	else:
		samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * np.convolve(samples_std, samples_std_2, mode)))
	return samples_corr