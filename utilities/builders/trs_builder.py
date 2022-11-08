from trsfile import trs_open
from trsfile import Header
from trsfile import TracePadding
from trsfile import Trace
from trsfile import SampleCoding
import numpy as np

class TRSBuilder:
	def __init__(self, path, description=None):
		#crea el directorio de guardado
		self.trs_file = trs_open(path, 'w',
					engine = 'TrsEngine',            # Optional: how the trace set is stored (defaults to TrsEngine)
					headers = {                      # Optional: headers (see Header class)
						Header.LABEL_X: 'points',
						Header.LABEL_Y: 'V',
						Header.DESCRIPTION: 'Traces' if description is None else description,
					},
					padding_mode = TracePadding.AUTO,# Optional: padding mode (defaults to TracePadding.AUTO)
					live_update = True               # Optional: updates the TRS file for live preview (small performance hit)
													 #   0 (False): Disabled (default)
													 #   1 (True) : TRS file updated after every trace
													 #   N        : TRS file is updated after N traces
													)
		self.__number_traces = 0
		self.__closed = False

	@property
	def Traces_Fed(self):
		return self.__number_traces
	
	def Feed_Traces(self, traces:np.ndarray, metadata:np.ndarray=None, title:str=None):
		self.trs_file.extend(Trace(SampleCoding.FLOAT, traces, 
							 data=metadata.tobytes() if metadata is not None else None, 
							 title=title if title is not None else ''))
		self.__number_traces += 1

	def __del__(self):
		if not self.__closed:
			self.Close_File()

	def Close_File(self):
		self.__closed = True
		self.trs_file.close()
