"""
@author: Servio Paguada
@email: serviopaguada@gmail.com
"""

from inspect import isclass
from typing import Tuple
import numpy as np
from sutils import trange
import os
import pickle
from sfileengine import H5FileEngine
#======================================================================================
def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
	"""Perform is_fitted validation for estimator.
	Checks if the estimator is fitted by verifying the presence of
	fitted attributes (ending with a trailing underscore) and otherwise
	raises a NotFittedError with the given message.
	This utility is meant to be used internally by estimators themselves,
	typically in their own predict / transform methods.
	
	:param estimator: Estimator instance for which the check is performed.
		it could be :class:`sklearn.preprocessing.StandardScaler` or 
		:class:`sklearn.preprocessing.MimMaxScaler`
	:type estimator: :class:`sklearn.preprocessing.StandardScaler`, or 
		:class:`sklearn.preprocessing.MimMaxScaler`
	
	:param attributes: Attribute name(s) given as string or a list/tuple of strings
		Eg.: ``["coef_", "estimator_", ...], "coef_"``
		If `None`, `estimator` is considered fitted if there exist an
		attribute that ends with a underscore and does not start with double
		underscore. 
		default=None
	:type attributes: :class:`str`, :class:`list` or tuple of :class:`str`, optional
	
	:param msg: The default error message is, "This %(name)s instance is not fitted
		yet. Call 'fit' with appropriate arguments before using this
		estimator."
		For custom messages if "%(name)s" is present in the message string,
		it is substituted for the estimator name.
		Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
		default=None
	:type msg: :class:`str`, optional
	
	:param all_or_any: Specify whether all or any of the given attributes must exist.
	:type all_or_any: callable, {all, any}, default=all
	
	:return: :class:`bool`
	:rtype: :class:`bool`, :class:`None`, Raise
	
	:except: :class:`NotFittedError`, If the attributes are not found.
	"""
	if isclass(estimator):
		raise TypeError("{} is a class, not an instance.".format(estimator))
	if msg is None:
		msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
			   "appropriate arguments before using this estimator.")

	if not hasattr(estimator, 'fit'):
		raise TypeError("%s is not an estimator instance." % (estimator))

	if attributes is not None:
		if not isinstance(attributes, (list, tuple)):
			attributes = [attributes]
		attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
	else:
		attrs = [v for v in vars(estimator)
				 if v.endswith("_") and not v.startswith("__")]

	if not attrs:
		return False
	return True
#======================================================================================
class DataBalancer:
	"""Represents a DataBalancer that creates a set of indexes of the traces by labels
	it uses the criterion of minimal class.

	Instancies a DataBalancer from a file that contains the information about itself

	:param filename: path of the data balancer file
	:type filename: :class:`str`	
	"""
	def __init__(self, filename:str):
		self.__filename = filename
	
	def get_minor_class(self) -> Tuple[int, int]:
		"""Gets the class which has the less number of traces related to it

		:return: A tuple of two int's
		:rtype: :class:`Tuple[int, int]`
		"""
		minor_class = None
		with open(self.__filename, 'rb') as handle:
			b = pickle.load(handle)
			key_minor = b['key_minor_value']
			minor_class = (key_minor, len(b['index_map'][key_minor]))
			
		return minor_class

	@staticmethod
	def get_from_dict(index_dict:dict, n_index, n_index_val=None) -> Tuple[dict, dict]:
		"""Gets the a tuple with two dicts that specifies the index of the array. 
		create_index_dict function outputs the dict requested by this function.

		:param n_index: Specifies the amount of indexes that we want.
		:type n_index: int

		:param n_index_val: Specifies the amount of indexes that we want to build a validation set, default to None
		:type n_index_val: int, optional
		
		:return: A tuple of two dicts, one for train set and the other for validation 
			set
		:rtype: Tuple[dict, dict]
		"""
		bd_dict = {}
		bd_val_dict = {}
		key_minor = index_dict['key_minor_value']
		minor_class = (key_minor, len(index_dict['index_map'][key_minor]))
		
		if n_index > minor_class[1]:
			print ('[WARNING]: Requested number of items is bigger than the smaller index:', n_index, minor_class, 'forcing balance')
			n_index = minor_class[1]

		index_map = index_dict['index_map']
		for key, value in index_map.items():
			bd_dict[key] = np.array(value, dtype=np.uint32)[:n_index]
			if n_index_val is not None:
				bd_val_dict[key] = np.array(value, dtype=np.uint32)[n_index:][:n_index_val]
		
		return (bd_dict, bd_val_dict)

			
	def get_data_balanced(self, n_index, n_index_val=None) -> Tuple[dict, dict]:
		"""Gets the a tuple with two dicts that specifies the index of the array
		taking care that it contains the same amount of indexes of the minor class.

		:param n_index: Specifies the amount of indexes that we want.
		:type n_index: int

		:param n_index_val: Specifies the amount of indexes that we want to build a validation set, default to None
		:type n_index_val: int, optional
		
		:return: A tuple of two dicts, one for train set and the other for validation 
			set
		:rtype: Tuple[dict, dict]
		"""
		bd_dict = {}
		bd_val_dict = {}
		with open(self.__filename, 'rb') as handle:
			b = pickle.load(handle)
			bd_dict, bd_val_dict = DataBalancer.get_from_dict(b, n_index, n_index_val)
		
		return (bd_dict, bd_val_dict)


	@staticmethod
	def create_index_dict(h5fileEngine:H5FileEngine) -> dict:
		"""Creates and data balacer base file from an H5FileEngine

		:param h5fileEngine: H5FileEngine of the trace set
		:type h5fileEngine: :class:`H5FileEngine`

		:param filename: the file name
		:type filename: :class:`str`

		:param suffix: Add a suffix to the filename
		:type suffix: `bool`

		:param path: the target path of the file
		:type path: :class:`str`
		"""
		class_couter = {}
		for i in trange(h5fileEngine.TotalTraces, desc='[INFO]: Processing balancer'):
			y = h5fileEngine[i][2]
			if y in class_couter.keys():
				class_couter[y].append(i)
			else:
				class_couter[y] = []
				class_couter[y].append(i)
		
		menor = np.inf
		llave_menor = np.inf
		for key, value in class_couter.items():
			if menor > len(value):
				menor = len(value)
				llave_menor = key
		
		result = {}
		result['key_minor_value'] = llave_menor
		result['index_map'] = class_couter

		return result
	
	@staticmethod
	def create_index_map(h5fileEngine:H5FileEngine, filename:str=None, suffix:bool=False, path:str=None) -> None:
		"""Creates and data balacer base file from an H5FileEngine

		:param h5fileEngine: H5FileEngine of the trace set
		:type h5fileEngine: :class:`H5FileEngine`

		:param filename: the file name
		:type filename: :class:`str`

		:param suffix: Add a suffix to the filename
		:type suffix: `bool`

		:param path: the target path of the file
		:type path: :class:`str`
		"""
		result = DataBalancer.create_index_dict(h5fileEngine)

		name = 'index_map.pickle'
		if filename is not None:
			name = filename if not suffix else 'index_map_{}.pickle'.format(filename)
		if path is not None:
			name = os.path.join(path, name)

		with open(name, 'wb') as handle:
			pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)