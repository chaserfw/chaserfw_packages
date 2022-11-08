"""Mean_Process

* author: Servio Paguada
* email: serviopaguada@gmail.com
"""

import numpy as np
from tqdm import trange

##################################################################################
#TRS support funtions
##################################################################################
def compute_samples_mean_trs(trs_file, n_traces, wtype=np.float64):
	# Get number of samples and metadata legth
	n_samples = len(trs_file[0])
	# Create resultant mean vector
	samples_mean = np.zeros(shape=(n_samples,), dtype=wtype)

	# Executing mean procedure
	for i in trange(n_traces, desc='[INFO *MeanProcess*]: Computing samples mean'):
		samples_mean = np.add(samples_mean, trs_file[i])
	
	samples_mean = np.true_divide(samples_mean, n_traces, dtype=np.float64)
	return samples_mean
##################################################################################
def compute_metadata_mean_trs(trs_file, n_traces, byte_list):
	# Get metadata vector dimension
	if byte_list is None:
		byte_list = range(len(np.frombuffer(trs_file[0].data, dtype=np.uint8)))
	metadata_dim = len(byte_list)

	# Create resultant mean vector
	metadata_mean = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	# Executing mean procedure
	for i in trange(n_traces, desc='[INFO *MeanProcess*]: Computing metadata mean'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_mean = np.add(metadata_mean, meta[byte_list])

	metadata_mean = np.true_divide(metadata_mean, n_traces, dtype=np.float64)
	return metadata_mean
##################################################################################
def compute_both_means_trs(trs_file, n_traces, byte_list):
	# Get number of samples and metadata legth
	n_samples = len(trs_file[0])

	# Get metadata vector dimension
	if byte_list is None:
		byte_list = range(len(np.frombuffer(trs_file[0].data, dtype=np.uint8)))
	metadata_dim = len(byte_list)

	# Create resultant mean vector
	samples_mean = np.zeros(shape=(n_samples,), dtype=np.float64)
	metadata_mean = np.zeros(shape=(metadata_dim,), dtype=np.float64)

	# Executing mean procedure
	#temp_metadata = np.empty(shape=(metadata_dim,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *MeanProcess*]: Computing both means'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_mean = np.add(metadata_mean, meta[byte_list])
		samples_mean = np.add(samples_mean, trs_file[i])

	samples_mean = np.true_divide(samples_mean, n_traces, dtype=np.float64)
	metadata_mean = np.true_divide(metadata_mean, n_traces, dtype=np.float64)
	return (samples_mean, metadata_mean)
##################################################################################
def compute_metadata_samples_mean_trs(trs_file, n_traces, byte_list=None):
	return compute_both_means_trs(trs_file, n_traces, byte_list)

##################################################################################
def compute_metadata_samples_mean_trs(trs_file, n_traces, byte_list=None):
	return compute_both_means_trs(trs_file, n_traces, byte_list)



##################################################################################
#TRS AES Sbox mean functions
##################################################################################
def compute_metadata_mean_trs_sbox(trs_file, plaintext_byte, key_byte, n_traces):
	from scryptoutils import AES_Sbox

	# Medatada dimension
	metadata_dim = 1

	# Create resultant mean vector
	metadata_mean = np.zeros(shape=(metadata_dim,), dtype=np.float64)

	# Executing mean procedure
	for i in trange(n_traces, desc='[INFO *MeanProcess-AESSboxBased*]: Computing metadata sbox mean'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_mean = np.add(metadata_mean, AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]])

	metadata_mean = np.true_divide(metadata_mean, n_traces, dtype=np.float64)
	return metadata_mean
##################################################################################
def compute_metadata_samples_mean_trs_sbox(trs_file, plaintext_byte, key_byte, n_traces):
	from scryptoutils import AES_Sbox

	# Get number of samples and metadata legth
	n_samples = len(trs_file[0])

	# Medatada dimension
	metadata_dim = 1

	# Create resultant mean vector
	samples_mean = np.zeros(shape=(n_samples,), dtype=np.float64)
	metadata_mean = np.zeros(shape=(metadata_dim,), dtype=np.float64)

	# Executing mean procedure
	for i in trange(n_traces, desc='[INFO *MeanProcess-AESSboxBased*]: Computing both means'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_mean = np.add(metadata_mean, AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]])
		samples_mean = np.add(samples_mean, trs_file[i])

	samples_mean = np.true_divide(samples_mean, n_traces, dtype=np.float64)
	metadata_mean = np.true_divide(metadata_mean, n_traces, dtype=np.float64)
	return (samples_mean, metadata_mean)

##################################################################################
#TRS Unmasked AES Sbox mean functions
##################################################################################
def compute_metadata_mean_trs_unmasked_sbox(trs_file, plaintext_byte, key_byte, mask_byte, n_traces):
	from scryptoutils import AES_Sbox

	# Medatada dimension
	metadata_dim = 1

	# Create resultant mean vector
	metadata_mean = np.zeros(shape=(metadata_dim,), dtype=np.float64)

	# Executing mean procedure
	for i in trange(n_traces, desc='[INFO *MeanProcess-UnmaskedAESSboxBased*]: Computing metadata unmasked sbox mean'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_mean = np.add(metadata_mean, AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] ^ meta[mask_byte])

	metadata_mean = np.true_divide(metadata_mean, n_traces, dtype=np.float64)
	return metadata_mean
##################################################################################
def compute_metadata_samples_mean_trs_unmasked_sbox(trs_file, plaintext_byte, key_byte, mask_byte, n_traces):
	from scryptoutils import AES_Sbox

	# Get number of samples and metadata legth
	n_samples = len(trs_file[0])

	# Medatada dimension
	metadata_dim = 1

	# Create resultant mean vector
	samples_mean = np.zeros(shape=(n_samples,), dtype=np.float64)
	metadata_mean = np.zeros(shape=(metadata_dim,), dtype=np.float64)

	# Executing mean procedure
	for i in trange(n_traces, desc='[INFO *MeanProcess-AESSboxBased*]: Computing both means'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_mean = np.add(metadata_mean, AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] ^ meta[mask_byte])
		samples_mean = np.add(samples_mean, trs_file[i])

	samples_mean = np.true_divide(samples_mean, n_traces, dtype=np.float64)
	metadata_mean = np.true_divide(metadata_mean, n_traces, dtype=np.float64)
	return (samples_mean, metadata_mean)