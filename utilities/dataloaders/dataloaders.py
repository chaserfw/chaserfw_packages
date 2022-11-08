"""
@author: Servio Paguada
@email: serviopaguada@gmail.com
"""
from sfileengine import h5fileengine
from sfileengine.h5fileengine import H5FileEngine
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import tables
from sutils import tqdm
from .utilities import check_is_fitted
from .utilities import DataBalancer
# ======================================================================================
#
# ======================================================================================


def load_dataset_sss(dataset_path, n_train_split, train_limit=None, n_val_split=0,
					 val_limit=0, scalers_list=None, samples=0, **kwargs):
	"""Loads data using StratifiedShuffleSplit methods.
	Balance the data to avoid class imbalance

	Args:
			dataset_path:
					The path where the dataset is storaged
			n_train_split:
					Number of examples of a class for the trainig set.
					Considering that there are 256 classes, meaning,
					the leakage model used is ID.
			train_limit:
					In the original dataset, the superior bound that limits the search of classes.
			n_val_split:
					Number of examples of a class for the validation set.
					Considering that there are 256 classes, meaning,
					the leakage model used is ID.
			val_limit:
					In the original dataset, the superior bound that limits the search of classes.
					val_limit works in conjuction with train_limit, creating a bounded interval
					[train_limit, val_limit]. It should be positive and bigger than train_limit
			scalers_list:
					A list of not fitted scalers.
			samples:
					The amount of samples taken to compound the new traces. If zero, it will take
					the whole samples. default (0)

	More args (kwargs):
			'metadata_type' : datatype of the metadata to set the target array
			'trace_type'	: datatype of the trace to set the target array
			'label_type'    : datatype of the label to set the target array

	Returns:
			A tuple comprises subtuple of (x_train, y_train, train_indexes), (x_val, y_val, val_indexes).
			If n_val_split parameter is 0, it returns only train tuple.
	"""

	metadata_type = np.uint8
	trace_type = np.float32
	label_type = np.uint8
	if 'metadata_type' in kwargs:
		metadata_type = kwargs['metadata_type']
	if 'trace_type' in kwargs:
		trace_type = kwargs['trace_type']
	if 'label_type' in kwargs:
		label_type = kwargs['label_type']

	file = tables.open_file(dataset_path)
	print('[INFO]: Profiling traces size:',
		  file.root.Profiling_traces.traces.shape)

	X_p = None
	Y_p = None
	maximun_limit = val_limit if val_limit != 0 else train_limit
	if samples > 0:
		X_p = np.array(
			file.root.Profiling_traces.traces[:maximun_limit, :samples], dtype=trace_type)
	else:
		X_p = np.array(
			file.root.Profiling_traces.traces[:maximun_limit, :], dtype=trace_type)

	Y_p = np.array(file.root.Profiling_traces.labels, dtype=label_type)

	if scalers_list is not None:
		print('[INFO]: Applying scalers')
		for scaler in scalers_list:
			X_p = scaler.fit_transform(X_p)

	X_profiling = np.empty(
		shape=(256*n_train_split, X_p.shape[1]), dtype=trace_type)
	Y_profiling = np.empty(shape=(256*n_train_split,), dtype=label_type)

	pro_indexes = np.empty(shape=(256*n_train_split,), dtype=np.uint32)

	shufflesplit = StratifiedShuffleSplit(
		n_splits=n_train_split, random_state=42, test_size=256)
	offset = 0

	train_limit = X_p.shape[0] if train_limit is None else train_limit

	pbar1 = tqdm(total=256*n_train_split, position=0,
				 desc='[INFO]: Building training set: ')
	for _, y in shufflesplit.split(X_p[:train_limit], Y_p[:train_limit]):
		X_profiling[offset:offset+256] = X_p[y]
		Y_profiling[offset:offset+256] = Y_p[y]
		pro_indexes[offset:offset+256] = y
		offset += 256
		pbar1.update()
	pbar1.close()
	if n_val_split != 0:
		X_validation = np.empty(
			shape=(256*n_val_split, X_p.shape[1]), dtype=trace_type)
		Y_validation = np.empty(shape=(256*n_val_split,), dtype=label_type)
		val_indexes = np.empty(shape=(256*n_val_split,), dtype=np.uint32)

		shufflesplit = StratifiedShuffleSplit(
			n_splits=n_val_split, random_state=42, test_size=256)
		offset = 0

		val_limit = X_p.shape[0] if val_limit == 0 else val_limit

		pbar2 = tqdm(total=256*n_val_split, position=0,
					 desc='[INFO]: Building validation set: ')
		for _, y in shufflesplit.split(X_p[train_limit:val_limit], Y_p[train_limit:val_limit]):
			X_validation[offset:offset+256] = X_p[y]
			Y_validation[offset:offset+256] = Y_p[y]
			val_indexes[offset:offset+256] = y
			offset += 256
			pbar2.update()
		pbar2.close()
	del X_p
	del Y_p

	print()
	print('[INFO]: Closing files')
	file.close()

	if n_val_split != 0:
		return (X_profiling, Y_profiling, pro_indexes), (X_validation, Y_validation, val_indexes)
	else:
		return (X_profiling, Y_profiling, pro_indexes)

# ======================================================================================
#
# ======================================================================================


def load_attack_dataset_sss(dataset_path, n_attack_split, attack_limit=None,
							scalers_list=None, samples=0, nbytes=16, **kwargs):
	"""Loads data from attack group using StratifiedShuffleSplit methods.
	Balance the data to avoid class imbalance

	Args:
			dataset_path:
					The path where the dataset is storaged
			n_attack_split:
					Number of examples of a class for the trainig set.
					Considering that there are 256 classes, meaning,
					the leakage model used is ID.
			attack_limit:
					In the original dataset, the superior bound that limits the search of classes.
			scalers_list:
					A list of not fitted scalers.
			samples:
					The amount of samples taken to compound the new traces. If zero, it will take
					the whole samples. default (0)
			nbytes:
					Number of bytes plaintext compresed

	More args (kwargs):
			'metadata_type' : datatype of the metadata to set the target array
			'trace_type'	: datatype of the trace to set the target array
			'label_type'    : datatype of the label to set the target array

	Returns:
			A tuple comprises subtuple of (x_train_attack, y_train_attack, plt_attack).
	"""

	metadata_type = np.uint8
	trace_type = np.float32
	label_type = np.uint8
	if 'metadata_type' in kwargs:
		metadata_type = kwargs['metadata_type']
	if 'trace_type' in kwargs:
		trace_type = kwargs['trace_type']
	if 'label_type' in kwargs:
		label_type = kwargs['label_type']

	file = tables.open_file(dataset_path)
	print('[INFO]: Attack traces size:', file.root.Attack_traces.traces.shape)

	X_p = None
	Y_p = None
	if samples > 0:
		X_p = np.array(file.root.Attack_traces.traces[:attack_limit, :samples], dtype=trace_type)
	else:
		X_p = np.array(file.root.Attack_traces.traces[:attack_limit, :], dtype=trace_type)

	Y_p = np.array(file.root.Attack_traces.labels, dtype=label_type)
	metadata = file.root.Attack_traces.metadata

	if scalers_list is not None:
		print('[INFO]: {} scalers'.format(
			'Fitting and applying' if not check_is_fitted(scalers_list[0]) else 'Applying'))
		for scaler in scalers_list:
			if check_is_fitted(scaler):
				X_p = scaler.transform(X_p)
			else:
				X_p = scaler.fit_transform(X_p)

	X_profiling = np.empty(shape=(256*n_attack_split, X_p.shape[1]), dtype=trace_type)
	Y_profiling = np.empty(shape=(256*n_attack_split,), dtype=label_type)
	plt_attack = np.empty(shape=(256*n_attack_split, nbytes), dtype=metadata_type)

	indexes = np.empty(shape=(256*n_attack_split,), dtype=np.uint32)

	shufflesplit = StratifiedShuffleSplit(n_splits=n_attack_split, random_state=42, test_size=256)
	offset = 0

	attack_limit = X_p.shape[0] if attack_limit is None else attack_limit

	pbar1 = tqdm(total=256*n_attack_split, position=0,
				 desc='[INFO]: Building training set: ')
	for _, y in shufflesplit.split(X_p[:attack_limit], Y_p[:attack_limit]):
		X_profiling[offset:offset+256] = X_p[y]
		Y_profiling[offset:offset+256] = Y_p[y]
		plt_attack[offset:offset + 256] = metadata.read_coordinates(y)['plaintext']
		indexes[offset:offset+256] = y
		offset += 256
		pbar1.update()
	pbar1.close()
	del X_p
	del Y_p
	print()
	print('[INFO]: Closing files')
	file.close()

	return (X_profiling, Y_profiling, plt_attack, indexes)

# ======================================================================================
#
# ======================================================================================


def load_dataset_sss_random(dataset_path, n_train_split, data_type=np.float32, train_limit=None,
							n_val_split=0, val_limit=0, scalers_list=None, samples=0):
	"""Loads data from proile group using StratifiedShuffleSplit methods from a computed
	permuted set of indexes. This function was created to increase the randomness when
	several experiments are aimed to be conducted. The natural follow step is to compute
	a confidence interval. This functions also serves as a way to apply randomness to a set
	of traces.

	Args:
			dataset_path:
					The path where the dataset is storaged
			n_train_split:
					Number of examples of a class for the trainig set.
					Considering that there are 256 classes, meaning,
					the leakage model used is ID.
			attack_limit:
					In the original dataset, the superior bound that limits the search of classes.
			scalers_list:
					A list of not fitted scalers.
			samples:
					The amount of samples taken to compound the new traces. If zero, it will take
					the whole samples. default (0)
			nbytes:
					Number of bytes plaintext compresed

	Returns:
			A tuple comprises subtuple of (x_train, y_train, train_indexes), (x_val, y_val, val_indexes).
			If n_val_split parameter is 0, it returns only train tuple.
	"""
	file = tables.open_file(dataset_path)
	print('[INFO]: Profiling traces size:',
		  file.root.Profiling_traces.traces.shape)

	print('[INFO]: Permutating traces')
	# Permutation to randomly choose traces
	ntraces = file.root.Profiling_traces.traces.shape[0]
	# Elaborates a permuted index vector
	permutation = np.random.permutation(ntraces)

	X_p = None
	Y_p = None
	maximun_limit = val_limit if val_limit != 0 else train_limit
	if samples > 0:
		X_p = np.array(
			file.root.Profiling_traces.traces[permutation[:maximun_limit], :samples], dtype=data_type)
	else:
		X_p = np.array(
			file.root.Profiling_traces.traces[permutation[:maximun_limit], :], dtype=data_type)

	# Takes the corresponding traces' labels by applying the permuted indexes
	Y_p = np.array(
		file.root.Profiling_traces.labels[permutation[:maximun_limit]], dtype=np.uint8)

	print('[INFO]: Applying scalers')
	if scalers_list is not None:
		for scaler in scalers_list:
			X_p = scaler.fit_transform(X_p)

	X_profiling = np.empty(
		shape=(256*n_train_split, X_p.shape[1]), dtype=np.float32)
	Y_profiling = np.empty(shape=(256*n_train_split,), dtype=np.uint8)

	shufflesplit = StratifiedShuffleSplit(
		n_splits=n_train_split, random_state=42, test_size=256)
	offset = 0

	train_limit = X_p.shape[0] if train_limit is None else train_limit

	# Define indication bar
	pbar1 = tqdm(total=256*n_train_split, position=0,
				 desc='[INFO]: Building training set: ')

	# Uses the StratifiedShuffleSplit to balance the classes,
	# applying the train_limits to avoid colliding with the validation set if any
	for X, y in shufflesplit.split(X_p[:train_limit], Y_p[:train_limit]):
		X_profiling[offset:offset+256] = X_p[y]
		Y_profiling[offset:offset+256] = Y_p[y]
		offset += 256
		pbar1.update(256)
	pbar1.close()
	# If a validation set is requested
	if n_val_split != 0:
		X_validation = np.empty(
			shape=(256*n_val_split, X_p.shape[1]), dtype=data_type)
		Y_validation = np.empty(shape=(256*n_val_split,), dtype=np.uint8)
		shufflesplit = StratifiedShuffleSplit(
			n_splits=n_val_split, random_state=42, test_size=256)
		offset = 0

		val_limit = X_p.shape[0] if val_limit == 0 else val_limit

		# Uses the StratifiedShuffleSplit to balance the classes,
		# applying the train_limits and val_limit to avoid colliding with the
		# training set if any
		pbar2 = tqdm(total=256*n_val_split, position=0,
					 desc='[INFO]: Building validation set: ')
		for _, y in shufflesplit.split(X_p[train_limit:val_limit], Y_p[train_limit:val_limit]):
			X_validation[offset:offset+256] = X_p[y]
			Y_validation[offset:offset+256] = Y_p[y]
			offset += 256
			pbar2.update(256)
		pbar2.close()
	del X_p
	del Y_p
	print()
	print('[INFO]: Closing files')
	file.close()

	if n_val_split != 0:
		return (X_profiling, Y_profiling, permutation[:train_limit]), (X_validation, Y_validation, permutation[train_limit:val_limit])
	else:
		return (X_profiling, Y_profiling, permutation[:train_limit])
# ======================================================================================
#
# ======================================================================================
def load_with_data_balancer(dataset_path, balancer_file, n_index, n_index_val=None,
							samples=0, scalers_list=None, **kwargs):

	metadata_type = np.uint8
	trace_type = np.float32
	label_type = np.uint8
	if 'metadata_type' in kwargs:
		metadata_type = kwargs['metadata_type']
	if 'trace_type' in kwargs:
		trace_type = kwargs['trace_type']
	if 'label_type' in kwargs:
		label_type = kwargs['label_type']

	db = DataBalancer(balancer_file)
	print('[INFO]: Minor class', db.get_minor_class())

	file = tables.open_file(dataset_path)
	print('[INFO]: Profiling traces size:', file.root.Profiling_traces.traces.shape)

	(db_dict, db_val_dict) = db.get_data_balanced(n_index, n_index_val)

	true_samples = file.root.Profiling_traces.traces.shape[1] if samples == 0 else samples
	X_profiling = np.empty(shape=(256*n_index, true_samples), dtype=trace_type)
	Y_profiling = np.empty(shape=(256*n_index,), dtype=label_type)
	pro_indexes = np.empty(shape=(256*n_index,), dtype=np.uint32)

	# Define indication bar
	pbar1 = tqdm(total=256*n_index, position=0, desc='[INFO]: Building training set: ')
	indexer = 0
	for _, value in db_dict.items():
		X_profiling[indexer:indexer+n_index] = file.root.Profiling_traces.traces[value, :true_samples]
		Y_profiling[indexer:indexer+n_index] = file.root.Profiling_traces.labels[value]
		pro_indexes[indexer:indexer+n_index] = value
		indexer += n_index
		pbar1.update(n_index)
	pbar1.close()

	if len(db_val_dict) != 0:
		X_validation = np.empty(shape=(256*n_index_val, true_samples), dtype=trace_type)
		Y_validation = np.empty(shape=(256*n_index_val,), dtype=label_type)
		val_indexes  = np.empty(shape=(256*n_index_val,), dtype=np.uint32)

		pbar2 = tqdm(total=256*n_index_val, position=0,
					 desc='[INFO]: Building validation set: ')
		indexer = 0
		for _, value in db_val_dict.items():
			X_validation[indexer:indexer+n_index_val] = file.root.Profiling_traces.traces[value, :true_samples]
			Y_validation[indexer:indexer+n_index_val] = file.root.Profiling_traces.labels[value]
			val_indexes[indexer:indexer+n_index_val]  = value
			indexer += n_index_val
			pbar2.update(n_index_val)
		pbar2.close()

	if scalers_list is not None:
		print('[INFO]: Applying scalers')
		for scaler in scalers_list:
			X_profiling = scaler.fit_transform(X_profiling)
			if len(db_val_dict) != 0:
				X_validation = scaler.transform(X_validation)
	file.close()
	if len(db_val_dict) != 0:
		return (X_profiling, Y_profiling, pro_indexes), (X_validation, Y_validation, val_indexes)
	else:
		return (X_profiling, Y_profiling, pro_indexes)
# ======================================================================================
#
# ======================================================================================
def load_attack_with_data_balancer(dataset_path, balancer_file, n_index,
								   samples=0, scalers_list=None, nbytes=16, **kwargs):
	"""Loads data from attack group using StratifiedShuffleSplit methods.
	Balance the data to avoid class imbalance

	Args:
		dataset_path:
			The path where the dataset is storaged
		balancer_file:
			Number of examples of a class for the trainig set.
			Considering that there are 256 classes, meaning,
			the leakage model used is ID.
		attack_limit:
			In the original dataset, the superior bound that limits the search of classes.
		scalers_list:
			A list of not fitted scalers.
		samples:
			The amount of samples taken to compound the new traces. If zero, it will take
			the whole samples. default (0)
		nbytes:
			Number of bytes plaintext compresed

	More args (kwargs):
		'metadata_type' : datatype of the metadata to set the target array
		'trace_type'	: datatype of the trace to set the target array
		'label_type'    : datatype of the label to set the target array

	Returns:
		A tuple comprises subtuple of (x_train_attack, y_train_attack, plt_attack).
	"""

	metadata_type = np.uint8
	trace_type = np.float32
	label_type = np.uint8
	if 'metadata_type' in kwargs:
		metadata_type = kwargs['metadata_type']
	if 'trace_type' in kwargs:
		trace_type = kwargs['trace_type']
	if 'label_type' in kwargs:
		label_type = kwargs['label_type']

	db = DataBalancer(balancer_file)
	minor_class = db.get_minor_class()
	print('[INFO]: Minor class', minor_class)

	file = tables.open_file(dataset_path)
	print('[INFO]: Attack traces size:',
		  file.root.Attack_traces.traces.shape)

	(db_dict, _) = db.get_data_balanced(n_index, None)
	if minor_class[1] < n_index:
		n_index = minor_class[1]

	true_samples = file.root.Attack_traces.traces.shape[1] if samples == 0 else samples
	X_profiling = np.empty(shape=(256*n_index, true_samples), dtype=trace_type)
	Y_profiling = np.empty(shape=(256*n_index,), dtype=label_type)
	pro_indexes = np.empty(shape=(256*n_index,), dtype=np.uint32)
	plt_attack = np.empty(shape=(256*n_index, nbytes), dtype=metadata_type)

	# Define indication bar
	pbar1 = tqdm(total=256*n_index, position=0,
				 desc='[INFO]: Building training set: ')

	indexer = 0
	for _, value in db_dict.items():
		X_profiling[indexer:indexer + n_index] = file.root.Attack_traces.traces[value, :true_samples]
		Y_profiling[indexer:indexer + n_index] = file.root.Attack_traces.labels[value]
		plt_attack[indexer:indexer  + n_index] = file.root.Attack_traces.metadata.read_coordinates(value)['plaintext']
		pro_indexes[indexer:indexer + n_index] = value
		indexer += n_index
		pbar1.update(n_index)
	pbar1.close()

	if scalers_list is not None:
		print('[INFO]: {} scalers'.format(
		'Fitting and applying' if not check_is_fitted(scalers_list[0]) else 'Applying'))
		for scaler in scalers_list:
			if check_is_fitted(scaler):
				X_profiling = scaler.transform(X_profiling)
			else:
				X_profiling = scaler.fit_transform(X_profiling)
	file.close()
	return (X_profiling, Y_profiling, plt_attack, pro_indexes)

# ======================================================================================
#
# ======================================================================================
def load_limit(dataset_path, train_limit:int, val_limit:int=0, scalers_list:list=None,
			   samples:int=0, **kwargs):
	"""Loads data from profiling group, from 0 to the specified train_limit.
	If val_limit specified it should be bigger than train_limit, and the returning
	entries would be from train_limit to val_limit

	Args:
		dataset_path (str):
				The path where the dataset is storaged
		train_limit (str):
				superior bound of the train set [0:train_limit]
		val_limit (int):
				superior bound of the validation set [train_limit:val_limit]
		scalers_list:
				A list of not fitted scalers.
		samples:
				The amount of samples taken to compound the new traces. If zero, it will take
				the whole samples. default (0)

	More args (kwargs):
			'metadata_type' : datatype of the metadata to set the target array
			'trace_type'	: datatype of the trace to set the target array
			'label_type'    : datatype of the label to set the target array

	Returns:
			A tuple comprises subtuple of (x_train, y_train, train_indexes), (x_val, y_val, val_indexes).
			If n_val_split parameter is 0, it returns only train tuple.
	"""

	metadata_type = np.uint8
	trace_type = np.float32
	label_type = np.uint8
	if 'metadata_type' in kwargs:
		metadata_type = kwargs['metadata_type']
	if 'trace_type' in kwargs:
		trace_type = kwargs['trace_type']
	if 'label_type' in kwargs:
		label_type = kwargs['label_type']

	file = tables.open_file(dataset_path)
	print('[INFO]: Profiling traces size:',
		  file.root.Profiling_traces.traces.shape)

	maximun_limit = val_limit if val_limit != 0 else train_limit
	__samples = samples if samples > 0 else file.root.Profiling_traces.traces.shape[1]
	X_p = np.array(
		file.root.Profiling_traces.traces[:maximun_limit, :__samples], dtype=trace_type)
	Y_p = np.array(
		file.root.Profiling_traces.labels[:maximun_limit], dtype=label_type)

	if scalers_list is not None:
		print('[INFO]: Fitting and applying scalers')
		for scaler in scalers_list:
			X_p = scaler.fit_transform(X_p)

	X_profiling = X_p[:train_limit]
	Y_profiling = Y_p[:train_limit]
	pro_indexes = np.array(range(train_limit))

	X_validation = None
	Y_validation = None
	if val_limit != 0:
		X_validation = X_p[train_limit:val_limit]
		Y_validation = Y_p[train_limit:val_limit]
		val_indexes = np.array(range(train_limit, val_limit))

	print()
	print('[INFO]: Closing files')
	file.close()

	if val_limit != 0:
		return (X_profiling, Y_profiling, pro_indexes), (X_validation, Y_validation, val_indexes)
	else:
		return (X_profiling, Y_profiling, pro_indexes)
# ======================================================================================
#
# ======================================================================================
def load_attack_limit(dataset_path, attack_limit:int, scalers_list=None, samples: int = 0,
					  **kwargs):
	"""Loads data from profiling group, from 0 to the specified train_limit.
	If val_limit specified it should be bigger than train_limit, and the returning
	entries would be from train_limit to val_limit

	Args:
		dataset_path (str):
			The path where the dataset is storaged
		attack_limit (int):
			superior bound of the attack set [0:attack_limit]
		scalers_list:
			A list of not fitted scalers.
		samples:
			The amount of samples taken to compound the new traces. If zero, it will take
					the whole samples. default (0)

	More args (kwargs):
			'metadata_type' : datatype of the metadata to set the target array
			'trace_type'	: datatype of the trace to set the target array
			'label_type'    : datatype of the label to set the target array

	Returns:
			A tuple comprises subtuple of (x_train_attack, y_train_attack, plt_attack).
	"""

	metadata_type = np.uint8
	trace_type = np.float32
	label_type = np.uint8
	if 'metadata_type' in kwargs:
		metadata_type = kwargs['metadata_type']
	if 'trace_type' in kwargs:
		trace_type = kwargs['trace_type']
	if 'label_type' in kwargs:
		label_type = kwargs['label_type']

	file = tables.open_file(dataset_path)
	__set_shape = file.root.Attack_traces.traces.shape
	print('[INFO]: Attack traces size:', __set_shape)
	if __set_shape[0] < attack_limit:
		print ('[WARNING]: Attack limit is bigger than the actual size of the dataset', __set_shape, attack_limit, 'forcing limit')
		attack_limit = __set_shape[0]

	__samples = samples if samples > 0 else file.root.Attack_traces.traces.shape[1]
	X_p = np.array(file.root.Attack_traces.traces[:attack_limit, :__samples], dtype=trace_type)
	Y_p = np.array(file.root.Attack_traces.labels[:attack_limit], dtype=label_type)
	
	if scalers_list is not None:
		print('[INFO]: {} scalers'.format('Fitting and applying' if not check_is_fitted(scalers_list[0]) else 'Applying'))
		for scaler in scalers_list:
			if check_is_fitted(scaler):
				X_p = scaler.transform(X_p)
			#else:
			#	X_p = scaler.fit_transform(X_p)

	X_profiling = X_p[:attack_limit]
	Y_profiling = Y_p[:attack_limit]
	plt_attack = file.root.Attack_traces.metadata.read_coordinates(np.array(range(attack_limit)))['plaintext']
	att_indexes = np.array(range(attack_limit))

	print()
	print('[INFO]: Closing files')
	file.close()

	return (X_profiling, Y_profiling, plt_attack, att_indexes)

# ======================================================================================
#
# ======================================================================================
def load_profiling_balanced(dataset_path, n_index, n_index_val=None, samples=0, 
							scalers_list=None, **kwargs):

	metadata_type = np.uint8
	trace_type = np.float32
	label_type = np.uint8
	if 'metadata_type' in kwargs:
		metadata_type = kwargs['metadata_type']
	if 'trace_type' in kwargs:
		trace_type = kwargs['trace_type']
	if 'label_type' in kwargs:
		label_type = kwargs['label_type']

	h5FileEngine = H5FileEngine(dataset_path, group='/Profiling_traces')

	print ('[INFO]: Computing index dict')
	db = DataBalancer.create_index_dict(h5FileEngine)
	print('[INFO]: Minor class', db['key_minor_value'])
	h5FileEngine.close()

	file = tables.open_file(dataset_path)
	print('[INFO]: Profiling traces size:', file.root.Profiling_traces.traces.shape)

	db_dict, db_val_dict = DataBalancer.get_from_dict(db, n_index, n_index_val)

	true_samples = file.root.Profiling_traces.traces.shape[1] if samples == 0 else samples
	X_profiling = np.empty(shape=(256*n_index, true_samples), dtype=trace_type)
	Y_profiling = np.empty(shape=(256*n_index,), dtype=label_type)
	pro_indexes = np.empty(shape=(256*n_index,), dtype=np.uint32)

	# Define indication bar
	pbar1 = tqdm(total=256*n_index, position=0, desc='[INFO]: Building training set: ')
	indexer = 0
	for _, value in db_dict.items():
		X_profiling[indexer:indexer+n_index] = file.root.Profiling_traces.traces[value, :true_samples]
		Y_profiling[indexer:indexer+n_index] = file.root.Profiling_traces.labels[value]
		pro_indexes[indexer:indexer+n_index] = value
		indexer += n_index
		pbar1.update(n_index)
	pbar1.close()

	if len(db_val_dict) != 0:
		X_validation = np.empty(shape=(256*n_index_val, true_samples), dtype=trace_type)
		Y_validation = np.empty(shape=(256*n_index_val,), dtype=label_type)
		val_indexes  = np.empty(shape=(256*n_index_val,), dtype=np.uint32)

		pbar2 = tqdm(total=256*n_index_val, position=0,
					 desc='[INFO]: Building validation set: ')
		indexer = 0
		for _, value in db_val_dict.items():
			X_validation[indexer:indexer+n_index_val] = file.root.Profiling_traces.traces[value, :true_samples]
			Y_validation[indexer:indexer+n_index_val] = file.root.Profiling_traces.labels[value]
			val_indexes[indexer:indexer+n_index_val]  = value
			indexer += n_index_val
			pbar2.update(n_index_val)
		pbar2.close()

	if scalers_list is not None:
		print('[INFO]: Fitting and applying scalers')
		for scaler in scalers_list:
			X_profiling = scaler.fit_transform(X_profiling)
			if len(db_val_dict) != 0:
				X_validation = scaler.transform(X_validation)
	file.close()
	if len(db_val_dict) != 0:
		return (X_profiling, Y_profiling, pro_indexes), (X_validation, Y_validation, val_indexes)
	else:
		return (X_profiling, Y_profiling, pro_indexes)
# ======================================================================================
#
# ======================================================================================
class AttackLoader():
	TypeLoader: str = 'Attack'
	LoaderMethod: str = ''

	def __init__(self) -> None:
		pass

	@staticmethod
	def with_balancer(dataset_path, balancer_file, n_index,
					  samples=0, scalers_list=None, nbytes=16, **kwargs):
		AttackLoader.LoaderMethod = 'with_balancer'
		return load_attack_with_data_balancer(dataset_path, balancer_file, n_index,
											  samples, scalers_list, nbytes, **kwargs)

	@staticmethod
	def with_limiters(dataset_path, attack_limit:int, scalers_list=None, samples:int=0,
					  **kwargs):
		"""Loads data from profiling group, from 0 to the specified train_limit.
			If val_limit specified it should be bigger than train_limit, and the returning
			entries would be from train_limit to val_limit

			Args:
				dataset_path (str):
					The path where the dataset is storaged
				attack_limit (str):
					superior bound of the attack set [0:train_limit]
				scalers_list:
					A list of not fitted scalers.
				samples:
					The amount of samples taken to compound the new traces. If zero, it will take
					the whole samples. default (0)

			More args (kwargs):
				'metadata_type' : datatype of the metadata to set the target array
				'trace_type'	: datatype of the trace to set the target array
				'label_type'    : datatype of the label to set the target array

			Returns:
				A tuple comprises subtuple of (x_train_attack, y_train_attack, plt_attack).
		"""
		AttackLoader.LoaderMethod = 'with_limiters'
		return load_attack_limit(dataset_path, attack_limit, scalers_list,
								 samples, **kwargs)
# ======================================================================================
#
# ======================================================================================

class ProfileLoader():
	TypeLoader = 'Profile'
	LoaderMethod: str = ''

	def __init__(self) -> None:
		pass

	@staticmethod
	def with_balancer(dataset_path, balancer_file, n_index, n_index_val=None,
					  samples=0, scalers_list=None, **kwargs):
		ProfileLoader.LoaderMethod = 'with_balancer'
		return load_with_data_balancer(dataset_path, balancer_file, n_index, n_index_val,
									   samples, scalers_list, **kwargs)

	@staticmethod
	def with_index_dict(dataset_path, n_index, n_index_val=None,
						samples=0, scalers_list=None, **kwargs):
		ProfileLoader.LoaderMethod = 'with_index_dict'
		return load_profiling_balanced(dataset_path, n_index, n_index_val,
									   samples, scalers_list, **kwargs)

	@staticmethod
	def with_limiters(dataset_path, train_limit:int, val_limit:int=0, scalers_list:list=None,
					  samples:int=0, **kwargs):
		"""Loads data from profiling group, from 0 to the specified train_limit.
		If val_limit specified it should be bigger than train_limit, and the returning
		entries would be from train_limit to val_limit

		Args:
			dataset_path (str):
					The path where the dataset is storaged
			train_limit (str):
					superior bound of the train set [0:train_limit]
			val_limit (int):
					superior bound of the validation set [train_limit:val_limit]
			scalers_list:
					A list of not fitted scalers.
			samples:
					The amount of samples taken to compound the new traces. If zero, it will take
					the whole samples. default (0)

		More args (kwargs):
			'metadata_type' : datatype of the metadata to set the target array
			'trace_type'	: datatype of the trace to set the target array
			'label_type'    : datatype of the label to set the target array

		Returns:
			A tuple comprises subtuple of (x_train, y_train, train_indexes), (x_val, y_val, val_indexes).
			If n_val_split parameter is 0, it returns only train tuple.
		"""
		ProfileLoader.LoaderMethod = 'with_limiters'
		return load_limit(dataset_path, train_limit, val_limit, scalers_list, samples,
						  **kwargs)
# ======================================================================================
#
# ======================================================================================
class DataLoader():
	AttackLoader = AttackLoader
	ProfileLoader = ProfileLoader

	def __init__(self) -> None:
		pass
