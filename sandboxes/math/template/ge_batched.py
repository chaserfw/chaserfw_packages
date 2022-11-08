from TA import TA_compute_guessing_entropy
import matplotlib.pyplot as plt
import tables
import numpy as np

covmatrix2 = np.load('cov_mean_matrix_2_covmatrix.npy')
meanMatrix = np.load('cov_mean_matrix_2_mean.npy')

dataset_path = 'ASCAD-compress.h5'

file        = tables.open_file(dataset_path, mode='r')
atkTraces   = file.root.Attack_traces.traces[:10000]
correct_key = file.root.Attack_traces.metadata[0]['key']
plt_attack  = file.root.Attack_traces.metadata[:10000]['plaintext']
print ('correct_key', correct_key)
print ('A plt_attack', plt_attack[0])
nb_traces = 10000
nb_attacks = 100
file.close()

avg_rank = TA_compute_guessing_entropy(atkTraces, plt_attack, meanMatrix, covmatrix2, correct_key[2], 
	nb_traces=700, nb_attacks=10, byte=2, shuffle=True)
print (avg_rank)

np.save('ge_test_5', avg_rank)
plt.plot(avg_rank, label='ta_'+str(32))
plt.legend()
plt.savefig('ge_test_5.pdf')
plt.show()

