import tables
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def scale(v):
	return (v - v.min()) / (v.max() - v.min())

def unscale(o, v):
	return o * (v.max() - v.min()) + v.min()



dataset_path = r'D:\SCA\datasets\ASCAD_var_desync100.h5'
file = tables.open_file(dataset_path, mode='r')
traces = file.root.Profiling_traces.traces[0:10000]

print ('scaled min', traces.min())
print (traces.min(axis=0))
print (min(traces.min(axis=0)))

mms = MinMaxScaler()
mim_traces = mms.fit_transform(traces)
scaled_traces = scale(traces)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(mim_traces[0])
ax2.plot(scaled_traces[0])
plt.show()