import numpy as np

class MetadataTypeDescriptor():
	def __init__(self):
		"""
		docstring
		"""
		self.__types = []

	def Add_Descriptor(self, set_name, numpy_type=np.uint8, nbytes=16):
		"""
		docstring
		"""
		if nbytes == 1:
			nbytes = (nbytes,)
		self.__types.append((set_name, numpy_type, nbytes))

	@property
	def Descriptor(self):
		"""
		docstring
		"""
		if len(self.__types) == 0:
			print ('[WARNING *MetadataTypeDescriptor*]: Descriptor list is empty, returning None')
			return None
		else:
			return np.dtype(self.__types)