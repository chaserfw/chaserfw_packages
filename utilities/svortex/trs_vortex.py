from .vortex import FileVortex
from .vortex import FormatConfig
import numpy as np
##################################################################################
class TRSFileVortex(FileVortex):
	def __init__(self, trs_file, input_format_config, otput_format_config, dtype=np.float):
		"""
		docstring
		"""
		super.__init__(trs_file, input_format_config, otput_format_config, dtype)
		
	def __len__(self):
		"""
		docstring
		"""
		return len(self._file_type)
	
	def __getitem__(self, key):
		if self._input_format_config == FormatConfig.FROM_ARRAY_TO_1DDLMODEL:
			trace = np.array(self._file_type[key], dtype=self._dtype)
			return trace.reshape(-1, 1)
		
		if self._input_format_config == FormatConfig.FROM_1DDLMODEL_TO_ARRAY:
			trace = np.array(self._file_type[key], dtype=self._dtype)
			return trace.reshape(-1, 1)
	
	def Close_File(self):
		self._trs_file.close()
##################################################################################
