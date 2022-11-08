import numpy as np

class Mixer():
	def __init__(self):
		pass
	def metadatamix(self, metadata_vector):
		pass


from .aes import AES_Sbox
class AESMaskedMixer(Mixex):
	def __init__(self, byte_1, byte_2):
		"""
		docstring
		"""
		Mixer.__init__(self)
		self.byte_1 = byte_1
		self.byte_2 = byte_2
	
	def metadatamix(self, metadata_vector):
        pass