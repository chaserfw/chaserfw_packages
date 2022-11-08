##################################################################################
# MEAN MODULE
##################################################################################
from .means_process import compute_samples_mean_trs
from .means_process import compute_metadata_mean_trs
from .means_process import compute_metadata_samples_mean_trs
from .means_process import compute_both_means_trs
from .means_process import compute_metadata_samples_mean_trs_unmasked_sbox
from .means_process import compute_metadata_mean_trs_unmasked_sbox
from .means_process import compute_metadata_samples_mean_trs_sbox
from .means_process import compute_metadata_mean_trs_sbox

##################################################################################
# FILE ENGINE PROCESS
##################################################################################
from .file_engine_process import compute_corr
from .file_engine_process import compute_corr_sbox
from .file_engine_process import compute_corr_unsbox
from .file_engine_process import compute_corr_double_unsbox

##################################################################################
# STANDARD DEVIATION MODULE
##################################################################################
from .stds_process import compute_samples_std_trs
from .stds_process import compute_metadata_std_trs
from .stds_process import compute_metadata_samples_std_trs
from .stds_process import compute_both_std_trs
from .stds_process import compute_metadata_std_trs_sbox
from .stds_process import compute_metadata_samples_std_trs_sbox
from .stds_process import compute_metadata_std_trs_unmasked_sbox
from .stds_process import compute_metadata_samples_std_trs_unmasked_sbox

##################################################################################
# CORRELATION MODULE
##################################################################################
from .corr_process import compute_metadata_traces_corr_trs
from .corr_process import compute_metadata_traces_corr_trs_sbox
from .corr_process import compute_metadata_traces_corr_trs_unmasked_sbox

##################################################################################
# ARITHMETICS
##################################################################################
from .aritmetics import remove_mean

##################################################################################
# SNR
##################################################################################
from .snr import snr_byte
from .snr import snr_var_mean
from .snr import snr_masked_sbox
from .snr import snr_unmasked_sbox

##################################################################################
# CONDIDENT INTERVAL
##################################################################################
from .confidence_interval import ge_confidence_mean

##################################################################################
# MUTUAL INFO
##################################################################################
from .mutual_info import shannon_entropy
from .mutual_info import calc_MI
from .mutual_info import sk_calc_MI
from .mutual_info import histo_based_entropy

##################################################################################
# SUMS OF DIFFERENCES
##################################################################################
from .ssod import SSOD
from .ssod import SSOD_from_fileengine

##################################################################################
# COVARIANCE
##################################################################################


##################################################################################
# ESTIMATORS MUTUAL INFO
##################################################################################
#from .mi_estimators import knn_based_entropy
#from .mi_estimators import ann_based_entropy

#from .mi_estimators import KLdivergence