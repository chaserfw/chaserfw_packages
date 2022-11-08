from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union
import numpy as np
from sfileengine import FileEngine
from scryptoutils import AES_Sbox
from sutils import trange
from sutils import tqdm
from dataloaders import DataBalancer
import random
# ======================================================================================
#
# ======================================================================================
def _sort_fengine_processor_plaintext_key(dataset:FileEngine, plaintext_pos:int, key_pos:int, 
										   ntraces=None, leakage_model='ID'):
	"""Creates a list of numpy arrays containing traces sorted according to the labels.
	
	:param dataset:
	:type dataset: :class:`FileEngine`

	:return: A numpy array comprises the index of the traces that belong to the specific class
	:rtype: :class:`numpy`
	"""
	ntraces = dataset.TotalTraces if ntraces is None else ntraces
	n_classes = 256 if leakage_model=='ID' else 9
	sorted_traceset = [[] for _ in range(n_classes)]
	for i in trange(ntraces, desc='[INFO *SortTraceSet*]: Clustering traces '):
		sorted_traceset[AES_Sbox[dataset[plaintext_pos] ^ dataset[key_pos]]].append(i)
	
	# Transform the list into a numpy array to easy further math
	return [np.array(sorted_traceset[i]) for i in range(n_classes)]
# ======================================================================================
#
# ======================================================================================
def _sort_fengine_processor_label(dataset:FileEngine, ntraces=None, leakage_model='ID'):
	"""Creates a list of numpy arrays containing traces sorted according to the labels.
	
	:param dataset:
	:type dataset: :class:`FileEngine`

	:return: A numpy array comprises the index of the traces that belong to the specific class
	:rtype: :class:`numpy`
	"""
	ntraces = dataset.TotalTraces if ntraces is None else ntraces
	n_classes = 256 if leakage_model=='ID' else 9
	sorted_traceset = [[] for _ in range(n_classes)]
	for i in trange(ntraces, desc='[INFO *SortTraceSet*]: Clustering traces '):
		sorted_traceset[dataset[i][2]].append(i)
	
	# Transform the list into a numpy array to easy further math
	return [np.array(sorted_traceset[i]) for i in range(n_classes)]
# ======================================================================================
#
# ======================================================================================
def sort_traceset(dataset:Union[FileEngine, np.ndarray, list], plaintext_pos:int=None, key_pos:int=None, 
				  ntraces=None, leakage_model='ID') -> np.ndarray:
	"""Creates a list of numpy arrays containing traces indexes sorted and grouped according to their label.

	:param dataset: Entity containing or at least from where labels can be drawn or created. 
		If dataset is type np.ndarray or list, it must contain the labels in the original order.
		If dataset is type FileEngine it is assumed that the returned format of the traces is:
		`[<trace>, [<metadata>, <label>]]`
	:type dataset: [:class:`FileEngine`, :class:`np.ndarray`, :class:`list`]

	:return: A numpy array comprises the index of the traces that belong to the specific class
	:rtype: :class:`np.ndarray`
	"""
	indexes_list = None
	if isinstance(dataset, FileEngine):
		if plaintext_pos is None or key_pos is None:
			indexes_list = _sort_fengine_processor_label(dataset, ntraces=ntraces, leakage_model=leakage_model)
		else:
			indexes_list = _sort_fengine_processor_plaintext_key(dataset, plaintext_pos, key_pos, ntraces, leakage_model)

	return np.array(indexes_list, dtype=np.ndarray)
# ======================================================================================
#
# ======================================================================================
def ID_wise_mean_balanced(
		dataset:FileEngine, balancer_file:str, n_indexes:int=None, 
		plaintext_pos:int=None, key_pos:int=None, by_label:bool=True, 
		clustered_traces:bool=False, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
	"""
	:param data_balancer: Provides to the function a balancer to use the same amount of traces per class
	:type data_balancer:
	"""

	m_seed = 42
	if 'seed' in kwargs:
		m_seed = kwargs['seed']
	
	m_n_classes = 256
	m_db = DataBalancer(balancer_file)
	m_minor_class = m_db.get_minor_class()
	print('[INFO *IDWiseMeanBalanced*]: Minor class: (key, value)', m_minor_class)
	n_indexes = m_minor_class[1] if n_indexes is None else n_indexes

	# Get the indexes dictionary
	(m_db_dict, _) = m_db.get_data_balanced(n_indexes)
	random.seed(m_seed)
	# Define indication bar
	pbar1 = tqdm(total=256*n_indexes, desc='[INFO *IDWiseMeanBalanced*]: Computing means')
	m_all_byte_slots = {}
	m_cluster_indexes = {} if clustered_traces else None
	for class_indexes in m_db_dict.values():
		random.shuffle(class_indexes)
		# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
		for trace_index in class_indexes[:n_indexes]:
			# Metadata vector
			metadata_vector = dataset[trace_index][1]
			
			# First time the byte appears in the dictionary
			selected_byte = AES_Sbox[metadata_vector[plaintext_pos] ^ metadata_vector[key_pos]] if not by_label else dataset[trace_index][2]
			if not (selected_byte in m_all_byte_slots):
				m_all_byte_slots[selected_byte] = StandardScaler()
				if m_cluster_indexes is not None:
					m_cluster_indexes[selected_byte] = np.zeros(0, dtype=np.uint32)
			
			# partially fit the scaler with the trace grouped by the key, increase by one the trace counter
			m_all_byte_slots[selected_byte].partial_fit(np.array(dataset[trace_index][0]).reshape(1, -1))
			if m_cluster_indexes is not None:
				m_cluster_indexes[selected_byte] = np.append(m_cluster_indexes[selected_byte], trace_index)
			
		pbar1.update(n_indexes)
	pbar1.close()

	# Matrix of all means and all vars, it's a list, which means that allows us to use 
	# with any crypto algorithm and not only 256 key byte algorithm	
	m_all_means  = np.zeros(0)
	m_all_vars   = np.zeros(0)
	m_mean_total = np.zeros(0)
	for _, scalers in m_all_byte_slots.items():
		m_all_means = np.append(m_all_means, scalers.mean_)
		m_all_vars  = np.append(m_all_vars, scalers.var_)

	m_all_means = np.reshape(m_all_means, (m_n_classes, dataset.TotalSamples))
	m_all_vars  = np.reshape(m_all_vars, (m_n_classes, dataset.TotalSamples))
	
	m_mean_total = np.mean(m_all_means, axis=0)

	return (m_mean_total, m_all_means, m_all_vars, m_cluster_indexes)
# ======================================================================================
#
# ======================================================================================
def ID_wise_mean(
		dataset:FileEngine, plaintext_pos:int=None, 
		key_pos:int=None, n_traces:int=None, 
		by_label:bool=True, clustered_traces:bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
	"""Computes operation-wise mean and all operations mean

	:param dataset:
	:type dataset: :class:`FileEngine`
	
	:param plaintext_pos:
	:type plaintext_pos: `int`
	
	:param key_pos:
	:type key_pos: `int`
	
	:param ntraces:
	:type ntraces: `int`
	
	:param by_label:
	:type by_label: `bool`

	:param clustered_traces: If true, it returns a dict comprises the indexes of traces beloging to an specific label, 
		if False, it returns None
	:type clustered_traces: `bool`

	:return: A tuple comprises mean_total, ID_op_mean, ID_op_variance
	:rtype: :class:`Tuple[np.ndarray, np.ndarray, np.ndarray, dict]`
	"""

	n_traces = dataset.TotalTraces if n_traces is None else n_traces
	m_n_classes = 256
	
	m_all_byte_slots = {}
	m_cluster_indexes = {} if clustered_traces else None
	# Compute all vector means using standard scaler, each vector represent the mean of a group formed from the given byte
	for i in trange(n_traces, desc='[INFO *IDWiseMean*]: Computing means'):
		# Metadata vector
		metadata_vector = dataset[i][1]
		
		# First time the byte appears in the dictionary
		selected_byte = AES_Sbox[metadata_vector[plaintext_pos] ^ metadata_vector[key_pos]] if not by_label else dataset[i][2]
		if not (selected_byte in m_all_byte_slots):
			m_all_byte_slots[selected_byte] = StandardScaler()
			if m_cluster_indexes is not None:
				m_cluster_indexes[selected_byte] = np.zeros(0, dtype=np.uint32)
		
		# partially fit the scaler with the trace grouped by the key, increase by one the trace counter
		m_all_byte_slots[selected_byte].partial_fit(np.array(dataset[i][0]).reshape(1, -1))
		if m_cluster_indexes is not None:
			m_cluster_indexes[selected_byte] = np.append(m_cluster_indexes[selected_byte], i)

	# Matrix of all means and all vars, it's a list, which means that allows us to use 
	# with any crypto algorithm and not only 256 key byte algorithm	
	m_all_means  = np.zeros(0)
	m_all_vars   = np.zeros(0)
	m_mean_total = np.zeros(0)
	for _, scalers in m_all_byte_slots.items():
		m_all_means = np.append(m_all_means, scalers.mean_)
		m_all_vars  = np.append(m_all_vars, scalers.var_)

	m_all_means = np.reshape(m_all_means, (m_n_classes, dataset.TotalSamples))
	m_all_vars  = np.reshape(m_all_vars, (m_n_classes, dataset.TotalSamples))
	
	m_mean_total = np.mean(m_all_means, axis=0)

	return (m_mean_total, m_all_means, m_all_vars, m_cluster_indexes)