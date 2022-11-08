from abc import abstractmethod, ABC
import numpy as np

def merging(metadata):
	merged = np.concatenate(list(metadata), axis=0)
	return merged.astype(np.uint8)

def abs(trace):
	return np.abs(trace)


class ProcessingEngine(ABC):
	def __init__(self) -> None:
		super().__init__()
	
	@abstractmethod
	def run(self, trace):
		pass

	@property
	def TotalTraces(self):
		return None
	
	@property
	def TotalSamples(self):
		return None

class Trimming(ProcessingEngine):
	def __init__(self, init_sample, end_sample) -> None:
		super().__init__()
		if init_sample >= end_sample:
			raise ValueError('[ERROR]: init_sample should be higher than end_sample')
		self._init_sample = init_sample
		self._end_sample = end_sample

	def run(self, trace):
		return trace[self._init_sample:self._end_sample]
	
	@property
	def TotalSamples(self):
		return self._end_sample - self._init_sample

class AbsTrimming(Trimming):
	def __init__(self, init_sample, end_sample) -> None:
		super().__init__(init_sample, end_sample)
	
	def run(self, trace):
		return np.abs(trace[self._init_sample:self._end_sample])

