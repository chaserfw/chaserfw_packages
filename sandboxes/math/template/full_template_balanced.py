from TA import TemplateAttack
from sfileengine import H5FileEngine
import numpy as np
import matplotlib.pyplot as plt
import tables

dataset_path = 'ASCAD-compress.h5'
balancer_file = r'..\..\sscametric-strainfunctions\ASCAD_dataset\index_map_ASCAD.pickle'
file = H5FileEngine(dataset_path, group='/Profiling_traces')

tAttack = TemplateAttack(file, balancer_file=balancer_file, n_indexes=None)
tAttack.compute_templates()
file.close()

file        = tables.open_file(dataset_path, mode='r')
atkTraces   = file.root.Attack_traces.traces[:10000]
correct_key = file.root.Attack_traces.metadata[0]['key']
plt_attack  = file.root.Attack_traces.metadata[:10000]['plaintext']
file.close()
print ('correct_key', correct_key)
print ('A plt_attack', plt_attack[0])
nb_traces = 600
nb_attacks = 1
avg_rank = tAttack.compute_ge(atkTraces, plt_attack, correct_key[2], 
	nb_traces=nb_traces, nb_attacks=nb_attacks, byte=2, shuffle=True)


np.save('ge_full_600', avg_rank)
plt.plot(avg_rank, label='ta_'+str(32))
plt.legend()
plt.savefig('ge_full_600.pdf')
plt.show()
