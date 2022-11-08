import os
import trsfile
import numpy as np
from .fileengine import FileEngine

class TRSFileEngine(FileEngine):
	def __init__(self, trs_file, mode='r', metadata_type=np.uint8):
		"""
		docstring
		"""	
		#FileEngine.__init__('s')
		if isinstance(trs_file, str):
			self._file_pointer = trsfile.open(trs_file, mode=mode)
			self._file         = trs_file
		else:	
			self._file_pointer = trs_file
			self._file         = trs_file.engine.path
		
		self.TracePointer      = 0
		self.__metadata_type   = metadata_type
		self._file_type 	   = 'trs'

	@property
	def MetadataTypeDescriptor(self):
		return None

	@property
	def TotalTraces(self):
		return len(self.TRSFile)

	@property
	def TotalSamples(self):
		return len(self.Trace[0])

	@property
	def TRSFile(self):
		"""
		docstring
		"""
		return self.FilePointer

	@property
	def Trace(self):
		"""
		docstring
		"""
		return [self.TRSFile[self.TracePointer], self.Data]
	
	@property
	def Data(self):
		return np.frombuffer(self.TRSFile[self.TracePointer].data, dtype=self.__metadata_type)

