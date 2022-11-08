from sscametric import perform_attacks
from templates import template_predict
from sfileengine import H5FileEngine
import numpy as np
import matplotlib.pyplot as plt
import tables
from scryptoutils import AES_Sbox
from sutils import trange
from scipy.stats import multivariate_normal
import random
import math

#======================================================================================
def rank_compute(att_pred, att_plt, mean_matrix, cov_matrix, byte, correct_key):
	m_P_k = np.zeros(256)
	m_nb_traces = att_pred.shape[0]
	m_rank_evol = np.full(m_nb_traces, 255)

	for j in trange(m_nb_traces, desc='[INFO]: Computing rank', position=1, leave=False):
		# Grab key points and put them in a matrix
		m_selected_trace = att_pred[j]
		# Test each key
		for k in range(256):
			# Find ID coming out of sbox
			ID = AES_Sbox[att_plt[j][byte] ^ k]
			# Find p_{k,j}
			rv = multivariate_normal(mean_matrix[ID], cov_matrix[ID])
			p_kj = rv.pdf(m_selected_trace)
			# Add it to running total
			m_P_k[k] += np.log(p_kj)
		
		# Sort the array in asc order, flip it, and then find the position of the greatest element
		m_final_rank = np.argwhere(np.flip(m_P_k.argsort()) == correct_key)[0][0]
		
		if math.isnan(float(m_final_rank)) or math.isinf(float(m_final_rank)):
			m_final_rank =  np.float32(256)
		else:
			m_final_rank = np.float32(m_final_rank)
		
		m_rank_evol[j] = m_final_rank

	return m_rank_evol

def template_mul(attack_traces, plt_attack, mean_matrix, cov_matrix, correct_key, nb_traces, nb_attacks=1, byte=2, shuffle=True):
	
	att_pred = None
	att_plt  = None
	all_rk_evol = np.zeros((nb_attacks, nb_traces))

	for attack in trange(nb_attacks, desc='[INFO]: Performing attack', position=0):
		if shuffle:
			l = list(zip(attack_traces, plt_attack))
			random.shuffle(l)
			sp, splt = list(zip(*l))
			sp = np.array(sp)
			splt = np.array(splt)
			att_pred = sp[:nb_traces]
			att_plt = splt[:nb_traces]

		else:
			att_pred = attack_traces[:nb_traces]
			att_plt = plt_attack[:nb_traces]
		
		all_rk_evol[attack] = rank_compute(att_pred, att_plt, mean_matrix, cov_matrix, byte, correct_key)	
	return np.mean(all_rk_evol, axis=0) 


covmatrix2 = np.load('cov_mean_matrix_covmatrix.npy')
meanMatrix = np.load('cov_mean_matrix_mean.npy')

dataset_path = 'ASCAD-compress.h5'
#file_attack = H5FileEngine(dataset_path, group='/Attack_traces')
#target_labels = np.array(file_attack[0:10000])[:,2]
#print ('some labels', target_labels[:10])
#predictions = template_predict(file_attack, mean_matrix=meanMatrix, cov_matrix=covmatrix2)
#file_attack.close()
#
#print (predictions.shape)
#print (predictions)

file        = tables.open_file(dataset_path, mode='r')
atkTraces   = file.root.Attack_traces.traces[:10000]
correct_key = file.root.Attack_traces.metadata[0]['key']
plt_attack  = file.root.Attack_traces.metadata[:10000]['plaintext']
print ('correct_key', correct_key)
print ('A plt_attack', plt_attack[0])
nb_traces = 10000
nb_attacks = 100
file.close()

avg_rank = template_mul(atkTraces, plt_attack, meanMatrix, covmatrix2, correct_key[2], nb_traces=700, nb_attacks=1, byte=2, shuffle=True)
print (avg_rank)
np.save('gessing_e_test', avg_rank)
plt.plot(avg_rank, label='ta_'+str(32))
plt.legend()
plt.savefig('ge_test_2.pdf')
plt.show()

#avg_rank = perform_attacks(nb_traces=nb_traces, predictions=predictions, 
#							 plt_attack=plt_attack, 
#							 correct_key=correct_key[2], 
#							 nb_attacks=nb_attacks, output_rank=True, pbar=True, byte=2)


#print (avg_rank)
#plt.plot(avg_rank, label='ta_'+str(32))
#plt.legend()
#plt.savefig('ge_test.pdf')
#plt.show()


""" Last result from the first try
[-323864.07351879 -327278.02298835 -326792.64091175 -330274.00225044
 -324057.588635   -327428.12391135 -326708.73322176 -330019.98244321
 -323973.80194193 -326824.25496447 -326972.06619577 -330346.58469466
 -324176.25905717 -327619.99576957 -326802.41456438 -330444.10170279
 -324466.98388534 -327899.01515261 -327214.61928437 -330726.25953825
 -323680.01778186 -328381.02367165 -327314.36477352 -330664.68051609
 -323795.17831863 -327538.72953048 -327076.72174648 -330737.9422488
 -324455.47769685 -328153.73048012 -326864.86047133 -330894.70064624
 -323782.85006089 -327156.11466166 -326573.4981517  -330582.85140904
 -323840.58284603 -327248.02217048 -326421.81779761 -330509.02052617
 -324047.14053794 -327140.8668724  -325961.92789182 -330130.11252813
 -323817.10428462 -327529.50475839 -326822.58492183 -329921.71330746
 -323882.52189698 -327550.24347828 -326627.23273528 -330496.81596345
 -323945.30904787 -327167.30637095 -327116.43895297 -330728.53072176
 -323879.00808286 -327593.41562211 -327156.0004479  -330262.36444297
 -324209.50356653 -328043.20204902 -327050.64001429 -330784.78363212
 -323716.95557193 -327298.04354257 -326324.59172759 -329842.86995436
 -323447.14607849 -327591.58987283 -326594.6455574  -331103.21488276
 -323901.88280244 -327442.18903838 -327202.69817092 -330168.05866725
 -323782.22311866 -327460.87288695 -326593.11711542 -330307.39359603
 -323993.12958418 -327372.53505702 -327128.26680794 -330588.38640013
 -323889.67378698 -327584.43626692 -327149.49350273 -330381.9250468
 -324095.12144397 -327518.73302915 -326381.37028879 -330639.21581292
 -324120.34000944 -328093.37046937 -327262.17389729 -330634.2117507
 -323523.12602412 -327191.1233923  -326593.17977877 -330123.56995179
 -323481.0616565  -327536.62173962 -326096.63413691 -330319.55377824
 -323231.11086057 -326925.51999669 -326599.68413868 -330054.10863793
 -323963.27825162 -327366.6889731  -326785.19199833 -329975.71871261
 -323746.70749232 -327111.81233202 -326218.9421513  -330689.93756262
 -324284.42905003 -327355.51277834 -326820.24136536 -330332.97851252
 -323532.26651027 -327687.00102849 -326967.08571103 -330384.54775259
 -323510.90810343 -327974.64792476 -327222.49133889 -330380.59056727
 -323350.08936214 -326932.86791423 -326183.66658916 -330073.83638036
 -323774.44016093 -327077.03968334 -326314.13659902 -330328.63638603
 -323938.83368253 -327272.77574032 -326989.46158442 -330319.43151896
 -324042.38210159 -327687.87910645 -327313.44027034 -329822.72332212
 -323504.33808816 -327324.29244103 -327119.20828546 -330052.01999886
 -324791.96997552 -327094.14382232 -326976.60095009 -330679.98919681
 -324683.22623604 -327461.46998299 -327312.98294105 -330054.15251617
 -324303.76557839 -327379.44528793 -327206.46097769 -330718.18777784
 -323270.29803836 -327338.78054419 -326303.88904681 -329838.72158891
 -323752.43832007 -327406.72253018 -326025.41242204 -330106.34273774
 -323695.19201906 -327280.19736261 -326982.42623083 -330063.63287444
 -324007.60495661 -327779.87502366 -326641.3994091  -330263.55930979
 -323675.92819667 -327285.6916223  -326669.06467125 -330703.19457292
 -324321.67161183 -327159.89203268 -326879.25929714 -330005.39580652
 -323228.0717313  -327602.95700052 -327118.48985152 -330333.62017085
 -324090.08918998 -327648.80848968 -327038.26225814 -330510.98904024
 -323252.71761668 -327726.52225534 -325832.00852831 -329730.4799492
 -323940.40524638 -327357.99256807 -326265.56509256 -330098.87687787
 -323223.78135538 -327522.05288437 -326229.12308005 -330280.25803189
 -323630.61931135 -327361.46025803 -327058.37173171 -330736.29469466
 -323312.49833796 -327073.87559211 -326551.69217248 -330305.86876338
 -324212.86484719 -327455.72232086 -326916.96366521 -330138.11651565
 -323625.53594227 -327685.3102957  -327134.74068463 -330713.90015669
 -324247.62944564 -327607.24242612 -327001.83823354 -331201.61952946
 -320905.18294765 -327103.06059226 -326397.29920961 -330034.44105704
 -323713.25944137 -326711.6532175  -326202.56991571 -330027.93694779
 -323704.22865534 -327340.09041444 -326221.46671785 -329862.413453
 -323641.79880924 -327338.43372437 -326121.9481991  -330044.5447464
 -324047.16152498 -327175.57531792 -327077.59165284 -330415.56387551
 -323887.49175933 -327514.61105834 -326590.99999327 -329919.34780817
 -323602.23127968 -327213.95942907 -326091.74438293 -330547.53429651
 -323833.4410691  -327505.95003905 -326680.62506254 -331265.72341641]
"""