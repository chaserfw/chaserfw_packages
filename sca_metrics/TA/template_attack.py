from templates import covmean_matrix
from templates import covmean_matrix_balanced
from .template_attack_ge import TA_compute_guessing_entropy
from sfileengine import H5FileEngine
from sutils import format_name
from typing import Union
import numpy as np
import os

class TemplateAttack():
	def __init__(
		self, train_fileengine:Union[H5FileEngine, str], train_dir:str='.', 
		balancer_file=None, n_indexes:int=None) -> None:

		if isinstance(train_fileengine, str):
			self.__train_fileengine = H5FileEngine(train_fileengine, group='/Profiling_traces')
		else:
			self.__train_fileengine = train_fileengine
		
		self.__n_indexes     = n_indexes
		self.__train_dir     = train_dir
		self.__balancer_file = balancer_file
		self.__mean_matrix   = None
		self.__cov_matrix    = None
		
	def compute_templates(self, save_templates=True):
		self.__mean_matrix, self.__cov_matrix = covmean_matrix(self.__train_fileengine) \
			if self.__balancer_file is None else \
			covmean_matrix_balanced(self.__train_fileengine, n_indexes=self.__n_indexes, balancer_file=self.__balancer_file)

		if save_templates:
			suffix = format_name()
			np.save(os.path.join(self.__train_dir, 'mean_matrix_{}'.format(suffix)), self.__mean_matrix)
			np.save(os.path.join(self.__train_dir, 'covmatrix_{}'.format(suffix)), self.__cov_matrix)
		
	def compute_ge(self, attack_traces:np.ndarray, plt_attack:np.ndarray, correct_key:int, nb_traces:int, 
                      nb_attacks:int=1, byte:int=2, shuffle:bool=True):
		"""Computes the guessing entropy for an SCA base template attack.

		:param attack_traces: The set of the whole traces from which the GE will be computed
		:type attack_traces: np.ndarray

		:param plt_attack: The set of the whole plaintext used to compute the intermediate value of the leakage model.
			In this computation, the leakage model is assumed to be ID; to work for HW leakage 
			model a modification is required
		:type plt_attack: np.ndarray

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

		return TA_compute_guessing_entropy(attack_traces, plt_attack, 
										   self.__mean_matrix, 
										   self.__cov_matrix, 
										   correct_key, 
										   nb_traces, 
                      					   nb_attacks, 
										   byte, 
										   shuffle)