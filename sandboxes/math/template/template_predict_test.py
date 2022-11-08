from templates import template_predict
from sfileengine import H5FileEngine
from sscametric import success_rate
import numpy as np
import matplotlib.pyplot as plt

covmatrix2 = np.load('cov_mean_matrix_2_covmatrix.npy')
meanMatrix = np.load('cov_mean_matrix_2_mean.npy')

dataset_path = 'ASCAD-compress.h5'
file_attack = H5FileEngine(dataset_path, group='/Attack_traces')
target_labels = np.array(file_attack[0:10000])[:,2]
result = template_predict(file_attack, mean_matrix=meanMatrix, cov_matrix=covmatrix2)
file_attack.close()
print (result.shape)
print (result)
span, c_sr = success_rate(result, target_labels=target_labels)
plt.plot(span, c_sr, label='ta_'+str(32))
plt.legend()
plt.show()
