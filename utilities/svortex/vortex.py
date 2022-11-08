from abc import ABC, abstractmethod
import enum
import numpy as np
##################################################################################
class FormatConfig(enum.Enum):
	FROM_ARRAY_TO_1DDLMODEL = 1
	FROM_1DDLMODEL_TO_ARRAY = 2
##################################################################################
class FileVortex(ABC):
	def __init__(self, file_type, input_format_config, otput_format_config, dtype=np.float):
		"""
		docstring
		"""
		self._input_format_config = input_format_config
		self._otput_format_config = otput_format_config
		self._file_type = file_type
		self._dtype = dtype
	
	@abstractmethod
	def __len__(self):
		"""
		docstring
		"""
		return
	
	@abstractmethod
	def __getitem__(self, key):
		"""
		docstring
		"""
		return
##################################################################################
class ArrayVortex():
	def __init__(self, format_config, dtype=np.float):
		"""
		docstring
		"""
		self._format_config = format_config
		self._dtype = dtype
	
	def Sinking(self, numpy_array):
		"""
		docstring
		"""
		if self._format_config == FormatConfig.FROM_ARRAY_TO_1DDLMODEL:
			return numpy_array.reshape(1, numpy_array.shape[0], 1)
		elif self._format_config == FormatConfig.FROM_1DDLMODEL_TO_ARRAY:
			return np.squeeze(numpy_array)