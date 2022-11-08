import numpy as np
from tqdm import trange

##################################################################################
# TRS support functions
##################################################################################
def compute_samples_std_trs(trs_file, n_traces, samples_mean):
	# Get number of samples and get metadata legth
	n_samples = len(trs_file[0])

	# Create resultant std vector
	samples_std  = np.zeros(shape=(n_samples,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *StdsProcess*]: Computing stdv'):
		samples_std  = np.add(samples_std,  np.power((trs_file[i] - samples_mean),  2))
	samples_std = np.sqrt((1/(n_traces-1)) * samples_std, dtype=np.float64)

	return samples_std
##################################################################################
def compute_metadata_std_trs(p_ascad_file, n_traces, metadata_mean, byte_list):
	# Get metadata vector dimension
	if byte_list is None:
		byte_list = range(len(np.frombuffer(trs_file[0].data, dtype=np.uint8)))
	metadata_dim = len(byte_list)

	# Create resultant std vector
	metadata_std = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *StdsProcess*]: Computing both stdv'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_std = np.add(metadata_std, np.power((meta[byte_list] - metadata_mean), 2))

	metadata_std = np.sqrt((1/(n_traces-1)) * metadata_std, dtype=np.float64)

	return metadata_std
##################################################################################
def compute_both_std_trs(trs_file, n_traces, samples_mean, metadata_mean, byte_list):
	# Get number of samples and get metadata legth
	n_samples = len(trs_file[0])
	# Get metadata vector dimension
	if byte_list is None:
		byte_list = range(len(np.frombuffer(trs_file[0].data, dtype=np.uint8)))
	metadata_dim = len(byte_list)

	# Create resultant std vector
	samples_std  = np.zeros(shape=(n_samples,),    dtype=np.float64)
	metadata_std = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *StdsProcess*]: Computing both stdv'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_std = np.add(metadata_std, np.power((meta[byte_list] - metadata_mean), 2))
		samples_std  = np.add(samples_std,  np.power((trs_file[i] - samples_mean),  2))

	metadata_std = np.sqrt((1/(n_traces-1)) * metadata_std, dtype=np.float64)
	samples_std = np.sqrt((1/(n_traces-1)) * samples_std, dtype=np.float64)

	return (samples_std, metadata_std)
##################################################################################
def compute_metadata_samples_std_trs(trs_file, n_traces, samples_mean, metadata_mean, byte_list=None):
	return compute_both_std_trs(trs_file, n_traces, samples_mean, metadata_mean, byte_list)

##################################################################################
#TRS AES Sbox mean functions
##################################################################################
def compute_metadata_std_trs_sbox(p_ascad_file, n_traces, metadata_mean, plaintext_byte, key_byte):
	# Import necessary packages
	from scryptoutils import AES_Sbox

	# Create resultant std vector
	metadata_std = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *StdsProcess-AESSboxBased*]: Computing both stdv'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_std = np.add(metadata_std, np.power((AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] - metadata_mean), 2))

	metadata_std = np.sqrt((1/(n_traces-1)) * metadata_std, dtype=np.float64)

	return metadata_std
##################################################################################
def compute_metadata_samples_std_trs_sbox(trs_file, samples_mean, metadata_mean, plaintext_byte, key_byte, n_traces):
	# Import necessary packages
	from scryptoutils import AES_Sbox

	# Medatada dimension
	metadata_dim = 1

	# Get number of samples and get metadata legth
	n_samples = len(trs_file[0])
	
	# Create resultant std vector
	samples_std  = np.zeros(shape=(n_samples,),    dtype=np.float64)
	metadata_std = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *StdsProcess-AESSboxBased*]: Computing both stdv'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_std = np.add(metadata_std, np.power((AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] - metadata_mean), 2))
		samples_std  = np.add(samples_std,  np.power((trs_file[i] - samples_mean),  2))

	metadata_std = np.sqrt((1/(n_traces-1)) * metadata_std, dtype=np.float64)
	samples_std = np.sqrt((1/(n_traces-1)) * samples_std, dtype=np.float64)

	return (samples_std, metadata_std)

##################################################################################
#TRS Unmasked AES Sbox mean functions
##################################################################################
def compute_metadata_std_trs_unmasked_sbox(p_ascad_file, metadata_mean, plaintext_byte, key_byte, mask_byte, n_traces):
	# Import necessary packages
	from scryptoutils import AES_Sbox

	# Medatada dimension
	metadata_dim = 1

	# Create resultant std vector
	metadata_std = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *StdsProcess-UnmaskedAESSboxBased*]: Computing both stdv'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_std = np.add(metadata_std, np.power((AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] ^ meta[mask_byte]) - metadata_mean, 2))

	metadata_std = np.sqrt((1/(n_traces-1)) * metadata_std, dtype=np.float64)

	return metadata_std
##################################################################################
def compute_metadata_samples_std_trs_unmasked_sbox(trs_file, samples_mean, metadata_mean, plaintext_byte, key_byte, mask_byte, n_traces):
	# Import necessary packages
	from scryptoutils import AES_Sbox

	# Medatada dimension
	metadata_dim = 1

	# Get number of samples and get metadata legth
	n_samples = len(trs_file[0])
	
	# Create resultant std vector
	samples_std  = np.zeros(shape=(n_samples,),    dtype=np.float64)
	metadata_std = np.zeros(shape=(metadata_dim,), dtype=np.float64)
	for i in trange(n_traces, desc='[INFO *StdsProcess-UnmaskedAESSboxBased*]: Computing both stdv'):
		meta = np.frombuffer(trs_file[i].data, dtype=np.uint8)
		metadata_std = np.add(metadata_std, np.power((AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] ^ meta[mask_byte]) - metadata_mean, 2))
		samples_std  = np.add(samples_std,  np.power((trs_file[i] - samples_mean),  2))

	metadata_std = np.sqrt((1/(n_traces-1)) * metadata_std, dtype=np.float64)
	samples_std = np.sqrt((1/(n_traces-1)) * samples_std, dtype=np.float64)

	return (samples_std, metadata_std)
