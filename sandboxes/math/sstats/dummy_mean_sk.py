from sstats import compute_samples_mean_trs
from sstats import compute_samples_std_trs
from sklearn.preprocessing import StandardScaler
import trsfile
from tqdm import trange 
import numpy as np

trs_file = trsfile.open(r'C:\Users\slpaguada\ChipWhisperer5_64\chipwhisperer\jupyter\Servio\traces_storaged\acq_20201008-141959\acq_20201008-141959.trs')
n_traces = len(trs_file)
mean = compute_samples_mean_trs(trs_file, n_traces)
print ('mean', mean)
std = compute_samples_std_trs(trs_file, n_traces, samples_mean=mean)
print ('std', std)

n_traces = len(trs_file)
sc = StandardScaler()
for i in trange(n_traces, desc=': Computing partial fit'):
    # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
    sc.partial_fit(np.array(trs_file[i]).reshape(1, -1))

print('skmean', sc.mean_)
print('skstd', np.sqrt(sc.var_))

assert sc.mean_.all() == mean.all()
assert np.sqrt(sc.var_).all() == std.all()

print ('mean_array', np.array(mean))
print ('mean_array_reshape',np.array(mean).reshape(1, -1))
print ('trs_array', np.array(trs_file[0]))
print ('trs_array .reshape(1, -1)', np.array(trs_file[0]).reshape(1, -1))
print ('trs_array .reshape(-1, 1)', np.array(trs_file[0]).reshape(-1, 1))
print ('transform reshape(1, -1)', sc.transform(np.array(trs_file[0]).reshape(1, -1)))
print ('transform reshape(-1, 1)', sc.transform(np.array(trs_file[0]).reshape(1, -1))[0] )
trs_file.close()