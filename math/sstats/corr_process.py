import numpy as np
from tqdm import trange
from sstats import compute_metadata_samples_mean_trs
from sstats import compute_metadata_samples_std_trs

##################################################################################
#  TRS functions
##################################################################################
def compute_corr_trs(trs_file, n_traces, metadata_byte, iterator, samples_mean, samples_std, metadata_mean, metadata_std):
    # Get group
    n_samples = len(trs_file[0])
    metadata_dim = 0    
    samples_corr    = np.zeros(shape=(n_samples,),    dtype=np.float64)
    single_metadata = np.zeros(shape=(metadata_dim,), dtype=np.float64)
    
    for i in trange(n_traces, desc='[INFO *CorrProcess*]: Computing correlation - byte: {}'.format(metadata_byte)):
        meta         = np.frombuffer(trs_file[i].data, dtype=np.uint8)
        samples_corr = np.add(samples_corr, (trs_file[i] - samples_mean) * (meta[metadata_byte] - metadata_mean[iterator]))
    if metadata_std[iterator] == 0:
        print ('[WARNING *CorrProcess*]: Metadata standard deviation of byte {} is zero'.format(metadata_byte))
        print ('[INFO *CorrProcess*]: Returning zero correlation')
    else:
        samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * metadata_std[iterator]))
    return samples_corr
##################################################################################
"""
def compute_metadata_traces_corr_trs(trs_file, n_traces, vb_list=[], samples_mean=None, samples_std=None):
    
    if (samples_mean is None) and (samples_std is None):
        (samples_mean, metadata_mean) = compute_metadata_samples_mean_trs (trs_file, n_traces, vb_list)
        (samples_std, metadata_std)   = compute_metadata_samples_std_trs  (trs_file, n_traces, samples_mean, metadata_mean, vb_list)
    
    samples_corr = []
    if len(vb_list) == 0:
        samples_corr.append(compute_corr_trs(trs_file, n_traces, 0, 0, samples_mean, samples_std, 
                            metadata_mean, metadata_std))
    else:
        for i, vb in enumerate(vb_list):
            samples_corr.append(compute_corr_trs(trs_file, n_traces, vb, i, samples_mean, samples_std, 
                                                    metadata_mean, metadata_std))
        
    return samples_corr
"""
##################################################################################
def compute_metadata_traces_corr_trs(file_engine, n_traces, vb_list=[], samples_mean=None, samples_std=None):
    
    if (samples_mean is None) and (samples_std is None):
        (samples_mean, metadata_mean) = compute_metadata_samples_mean_trs (trs_file, n_traces, vb_list)
        (samples_std, metadata_std)   = compute_metadata_samples_std_trs  (trs_file, n_traces, samples_mean, metadata_mean, vb_list)
    
    samples_corr = []
    if len(vb_list) == 0:
        samples_corr.append(compute_corr_trs(trs_file, n_traces, 0, 0, samples_mean, samples_std, 
                            metadata_mean, metadata_std))
    else:
        for i, vb in enumerate(vb_list):
            samples_corr.append(compute_corr_trs(trs_file, n_traces, vb, i, samples_mean, samples_std, 
                                                    metadata_mean, metadata_std))
        
    return samples_corr

##################################################################################
#  TRS AES Sbox function based correlation procedures
##################################################################################
def compute_corr_trs_sbox(trs_file, n_traces, plaintext_byte, key_byte, samples_mean, samples_std, metadata_mean, metadata_std):
    """Correlation based on AES Sbox
    """
    from scryptoutils import AES_Sbox

    # Get group
    n_samples = len(trs_file[0])
    metadata_dim = 0    
    samples_corr    = np.zeros(shape=(n_samples,),    dtype=np.float64)
    single_metadata = np.zeros(shape=(metadata_dim,), dtype=np.float64)
    
    for i in trange(n_traces, desc='[INFO *CorrProcess-AESSboxBased*]: Computing correlation plaintext {} and key {}'.format(plaintext_byte, key_byte)):
        meta         = np.frombuffer(trs_file[i].data, dtype=np.uint8)
        samples_corr = np.add(samples_corr, (trs_file[i] - samples_mean) * (AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] - metadata_mean))
        
    if np.count_nonzero(metadata_std) == 0 or np.count_nonzero(samples_std) == 0:
        print ('[WARNING *CorrProcess-AESSboxBased*]: Metadata or samples standard deviation of AES Sbox plaintext {} and key {} is zero'.format(plaintext_byte, key_byte))
        print ('[INFO *CorrProcess-AESSboxBased*]: Returning zero correlation')
    else:
        print ('stds', samples_std)
        print ('mean', metadata_std)
        samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * metadata_std))
    return samples_corr
##################################################################################
def compute_metadata_traces_corr_trs_sbox(trs_file, n_traces, plaintext_byte, key_byte, samples_mean=None, samples_std=None):
    from sstats import compute_metadata_samples_std_trs_sbox
    from sstats import compute_metadata_samples_mean_trs_sbox

    if (samples_mean is None) and (samples_std is None):
        (samples_mean, metadata_mean) = compute_metadata_samples_mean_trs_sbox (trs_file, plaintext_byte, key_byte, n_traces)
        (samples_std, metadata_std)   = compute_metadata_samples_std_trs_sbox  (trs_file, samples_mean, metadata_mean, 
                                                                                plaintext_byte, key_byte, n_traces)
    
    samples_corr = [compute_corr_trs_sbox(trs_file, n_traces, plaintext_byte, key_byte, samples_mean, samples_std, metadata_mean, metadata_std)]
        
    return samples_corr

##################################################################################
#  TRS Unmasked AES Sbox function based correlation procedures
##################################################################################
def compute_corr_trs_unmasked_sbox(trs_file, n_traces, plaintext_byte, key_byte, mask_byte, samples_mean, samples_std, metadata_mean, metadata_std):
    """Correlation based on unmasked AES Sbox
    """
    from scryptoutils import AES_Sbox

    # Get group
    n_samples = len(trs_file[0])
    metadata_dim = 0    
    samples_corr    = np.zeros(shape=(n_samples,),    dtype=np.float64)
    single_metadata = np.zeros(shape=(metadata_dim,), dtype=np.float64)
    
    for i in trange(n_traces, desc='[INFO *CorrProcess-UnmaskedAESSboxBased*]: Computing correlation AES-Sbox([{}]^[{}])^[{}]'.format(plaintext_byte, key_byte, mask_byte)):
        meta         = np.frombuffer(trs_file[i].data, dtype=np.uint8)
        samples_corr = np.add(samples_corr, (trs_file[i] - samples_mean) * ((AES_Sbox[meta[plaintext_byte] ^ meta[key_byte]] ^ meta[mask_byte]) - metadata_mean))
    if np.count_nonzero(metadata_std) == 0 or np.count_nonzero(samples_std) == 0:
        print ('[WARNING *CorrProcess-UnmaskedAESSboxBased*]: Metadata standard deviation of unmasked AES Sbox plaintext {}, key {} and mask {} is zero'.format(plaintext_byte, key_byte, mask_byte))
        print ('[INFO *CorrProcess-UnmaskedAESSboxBased*]: Returning zero correlation')
    else:
        samples_corr = np.true_divide(samples_corr, ((n_traces - 1) * samples_std * metadata_std))
    return samples_corr
##################################################################################
def compute_metadata_traces_corr_trs_unmasked_sbox(trs_file, n_traces, plaintext_byte, key_byte, mask_byte, samples_mean=None, samples_std=None):
    from sstats import compute_metadata_samples_mean_trs_unmasked_sbox
    from sstats import compute_metadata_samples_std_trs_unmasked_sbox

    if (samples_mean is None) and (samples_std is None):
        (samples_mean, metadata_mean) = compute_metadata_samples_mean_trs_unmasked_sbox (trs_file, plaintext_byte, key_byte, mask_byte, n_traces)
        (samples_std, metadata_std)   = compute_metadata_samples_std_trs_unmasked_sbox  (trs_file, samples_mean, metadata_mean, 
                                                                                         plaintext_byte, key_byte, mask_byte, n_traces)
    
    samples_corr = [compute_corr_trs_unmasked_sbox(trs_file, n_traces, plaintext_byte, key_byte, mask_byte, samples_mean, samples_std, metadata_mean, metadata_std)]
        
    return samples_corr