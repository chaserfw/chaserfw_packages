import os
import trsfile
import numpy as np
import os
from .fileengine import FileEngine

class MemoryFileEngine(FileEngine):
	def __init__(self, traces, metadata=None, label=None, processing_function=None):
		""" Instanciate a file engine from memory allocated data
		Args:
			:param traces: A 2D numpy array or table arrray whose elements are the traces.
			:type traces: numpy, tables.array

			:param metadata: A 2D numpy array or table arrray/table whose elements are metadata vector.
			:type metadata: numpy, tables.array

			:param label: A 2D numpy array or table arrray whose elements are label of the traces.
			:type label: numpy, tables.array

		"""
		super().__init__()

		self.__label    = label
		self.__metadata = metadata
		self.__traces   = traces
		
		self._file_type    = 'memory'
		self._file_pointer = None

		if processing_function is not None and isinstance(processing_function, str):
			if processing_function == 'merging':
				from .processors import merging
				processing_function = merging

		self.__processing_function = processing_function if processing_function is not None else self.__identity_preprocesing

	def __identity_preprocesing(self, data):
		return data

	@property
	def MetadataTypeDescriptor(self):
		if self.__metadata is not None:
			return self.__metadata.dtype
		else:
			return None
		
	@property
	def TotalTraces(self):
		return self.__traces.shape[0]

	@property
	def TotalSamples(self):
		return self.__traces.shape[1]

	@property
	def OriginalPath(self):
		return '.{}'.format(os.sep)
	
	@property
	def Label(self):
		return self.__label[self.TracePointer]

	@property
	def Trace(self):
		"""
		docstring
		"""
		if self.__label is None:
			return [self.__traces[self.TracePointer], self.Data]
			#return np.array([(self.__traces[self.TracePointer], self.Data)], 
			#dtype=([('trace', np.ndarray), ('meta', np.ndarray)]))
			#return np.rec.fromarrays((self.__traces[self.TracePointer], self.Data), 
			#names=('trace', 'meta'))
			#return np.array([self.__traces[self.TracePointer], self.Data], dtype=np.ndarray)
		else:
			return [self.__traces[self.TracePointer], self.Data, self.Label]
	
	@property
	def Data(self):	
		#return np.concatenate(list(self.__metadata[self.TracePointer]), axis=0)
		if self.__metadata is not None:
			return self.__processing_function(self.__metadata[self.TracePointer])
		else:
			return None
	
