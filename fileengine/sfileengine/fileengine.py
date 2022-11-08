import os
import trsfile
import numpy as np
from abc import abstractmethod, ABC

class FileEngine(ABC):
	def __init__(self):
		"""
		docstring
		"""	
		self._file         = None
		self.__trace_pointer = 0
		self._file_pointer = None
		self._file_type    = None
	
	@property
	@abstractmethod
	def MetadataTypeDescriptor(self):
		pass
		
	@property
	@abstractmethod
	def TotalTraces(self) -> int:
		pass

	@property
	@abstractmethod
	def TotalSamples(self) -> int:
		pass

	@property
	@abstractmethod
	def Trace(self):
		pass
	
	@property
	@abstractmethod
	def Data(self):
		pass

	def __len__(self) -> int:
		"""Returns the total number of samples in this trace"""
		return self.TotalTraces
	
	def __getitem__(self, index):	
		if isinstance(index, slice):
			#Get the start, stop, and step from the slice
			return [self[ii] for ii in range(*index.indices(len(self)))]
		elif isinstance(index, list) or isinstance(index, np.ndarray):
			# return traces using the indexes coming in index variable
			# index should be an 1D array
			return [self[ii] for ii in index]
		else:
			self.TracePointer = index
			return self.Trace

	def __iter__(self):
		""" reset pointer """
		self.__trace_pointer = -1
		return self

	def __next__(self):
		self.Next
		if self.__trace_pointer >= len(self):
			raise StopIteration
		return self[self.__trace_pointer]

	@property
	def OriginalPath(self) -> str:
		return os.path.basename(self._file)

	@property
	def FileExtension(self):
		return os.path.splitext(os.path.basename(self.OriginalPath))[1]

	@property
	def FileName(self):
		return os.path.splitext(os.path.basename(self.OriginalPath))[0]

	@property
	def Path(self) -> str:
		"""
		docstring
		"""
		return os.path.dirname(os.path.abspath(self.OriginalPath))	

	@property
	def FullPath(self):
		"""
		docstring
		"""
		return os.path.abspath(self.OriginalPath)

	@property
	def FilePointer(self):
		"""
		docstring
		"""
		return self._file_pointer

	@property
	def TracePointer(self):
		"""
		docstring
		"""
		return self.__trace_pointer
	
	@TracePointer.setter
	def TracePointer(self, trace_pointer):
		"""
		docstring
		"""
		if trace_pointer >= self.TotalTraces:
			self.__trace_pointer = self.TotalTraces - 1
		elif trace_pointer < 0:
			self.__trace_pointer = 0
		else:
			self.__trace_pointer = trace_pointer
	
	@property
	def NextTrace(self):
		"""
		docstring
		"""
		self.Next
		return self.Trace
	
	@property
	def PreviousTrace(self):
		"""
		docstring
		"""
		self.Previous
		return self.Trace
	
	@property
	def Next(self):
		"""
		docstring
		"""
		self.TracePointer += 1
	
	@property
	def Previous(self):
		"""
		docstring
		"""
		self.TracePointer -= 1
	
	@property
	def shape(self):
		return (self.TotalTraces, self.TotalSamples)

	@property
	def FileType(self):
		return self._file_type

	def __del__(self):
		self.close()

	def close(self):
		if self.FilePointer is not None:
			self.FilePointer.close()