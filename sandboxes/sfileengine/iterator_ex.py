from sfileengine import TRSFileEngine
from sfileengine import H5FileEngine
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from sstats import snr_byte

#path = r'D:\ASCAD_EPF_20201222-105722.trs'
#path = r'C:\Users\slpaguada\Datasets\ASCAD\ATMEGA_AES_v1\ATM_AES_v1_variable_key\ASCAD_data\ASCAD_databases\ASCAD_var.h5'
path = r'D:\SCA\datasets\ASCAD_var_desync100.h5'

#file = TRSFileEngine(path)
file = H5FileEngine(path, group="/Profiling_traces")

print (file.TotalTraces)
print (file.TotalSamples)
print (file.OriginalPath)
print (file.FileName)
print (file.Path)

print (file.FileType)
print (file[0])
print (len(file))

for i in trange(10):
	print (file[i][0][0])


snr = snr_byte(file, [18])
print(snr)

time = np.linspace(0, 1400, 1400)
plt.plot(time, snr[0][0])
plt.show()

file.close()






"""
class TeamIterator:

	''' Iterator class '''
	def __init__(self, team):
		# Team object reference
		self._team = team
		# member variable to keep track of current index
		self._index = 0
	def __next__(self):
		''''Returns the next value from team object's lists '''
		if self._index < (len(self._team._juniorMembers) + len(self._team._seniorMembers)) :
			if self._index < len(self._team._juniorMembers): # Check if junior members are fully iterated or not
				result = (self._team._juniorMembers[self._index] , 'junior')
			else:
				result = (self._team._seniorMembers[self._index - len(self._team._juniorMembers)]   , 'senior')
			self._index +=1
			return result
		# End of Iteration
		raise StopIteration


class Team:
	def __init__(self):
		self.__list = ['df', '45']
	def __iter__(self):
		''' Returns the Iterator object '''
		return TeamIterator(self)
"""