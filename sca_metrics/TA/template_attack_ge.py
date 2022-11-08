from scryptoutils import AES_Sbox
from sutils import trange
from scipy.stats import multivariate_normal
import random
import math
import numpy as np
#======================================================================================
def __rank_compute(att_pred, att_plt, mean_matrix, cov_matrix, byte, correct_key):
	m_P_k = np.zeros(256)
	m_nb_traces = att_pred.shape[0]
	m_rank_evol = np.full(m_nb_traces, 255)

	for j in trange(m_nb_traces, desc='[INFO]: Computing rank', position=1, leave=False):
		# Grab key points and put them in a matrix
		m_selected_trace = att_pred[j]
		# Test each key
		for k in range(256):
			# Find ID coming out of sbox
			ID = AES_Sbox[att_plt[j][byte] ^ k]
			# Find p_{k,j}
			rv = multivariate_normal(mean_matrix[ID], cov_matrix[ID])
			p_kj = rv.pdf(m_selected_trace)
			# Add it to running total
			m_P_k[k] += np.log(p_kj)
		
		# Sort the array in asc order, flip it, and then find the position of the greatest element
		m_final_rank = np.argwhere(np.flip(m_P_k.argsort()) == correct_key)[0][0]
		
		if math.isnan(float(m_final_rank)) or math.isinf(float(m_final_rank)):
			m_final_rank =  np.float32(256)
		else:
			m_final_rank = np.float32(m_final_rank)
		
		m_rank_evol[j] = m_final_rank

	return m_rank_evol

def TA_compute_guessing_entropy(attack_traces:np.ndarray, plt_attack:np.ndarray, mean_matrix:np.ndarray, 
					  cov_matrix:np.ndarray, correct_key:int, nb_traces:int, 
                      nb_attacks:int=1, byte:int=2, shuffle:bool=True):
	"""Computes the guessing entropy for an SCA base template attack.

	:param attack_traces: The set of the whole traces from which the GE will be computed
	:type attack_traces: np.ndarray

	:param plt_attack: The set of the whole plaintext used to compute the intermediate value of the leakage model.
		In this computation, the leakage model is assumed to be ID; to work for HW leakage 
		model a modification is required
	:type plt_attack: np.ndarray

	:param mean_matrix: The mean matrix that refers to the template mean matrix in a template attack
	:type mean_matrix: np.ndarray

	:param cov_matrix: Likewise, the covariance matrix that refers to the template covariance matrix in a template attack
	:type cov_matrix: np.ndarray

	:param correct_key: The value of the correct key byte
	:type correct_key: int

	:param nb_traces: Number of attack traces, it will compound a set or a subset of the whole traceset depending
		on the specified number of traces. 
	:type nb_traces: int

	:param nb_attacks: The number of times the guessing entropy will be computed, if nb_attacks > 1 an average 
		will be returned.
	:type nb_attacks: int

	:param byte: The position of the correct key byte aimed to be recover
	:type byte: int

	:param shuffle: if True the set of subset of traces (specified by the value of the nb_traces parameter) will
		be shuffled. The shuffle happens as many times as nb_attacks defines.
	:type shuffle: bool
	"""
	att_pred = None
	att_plt  = None
	all_rk_evol = np.zeros((nb_attacks, nb_traces))

	for attack in trange(nb_attacks, desc='[INFO]: Computing GE', position=0):
		if shuffle:
			l = list(zip(attack_traces, plt_attack))
			random.shuffle(l)
			sp, splt = list(zip(*l))
			sp = np.array(sp)
			splt = np.array(splt)
			att_pred = sp[:nb_traces]
			att_plt = splt[:nb_traces]

		else:
			att_pred = attack_traces[:nb_traces]
			att_plt = plt_attack[:nb_traces]
		
		all_rk_evol[attack] = __rank_compute(att_pred, att_plt, mean_matrix, cov_matrix, byte, correct_key)	
	return np.mean(all_rk_evol, axis=0)