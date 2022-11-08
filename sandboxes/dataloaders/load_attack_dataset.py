from dataloaders import load_attack_dataset_sss
from dataloaders import load_dataset_sss
from dataloaders import load_dataset_sss_random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

print ('===============================load_attack_dataset_sss')
scalers_list = [StandardScaler(), MinMaxScaler()]
dataset_path = r'D:\AES_PT\AES_PT_masked-D1-label-0.h5'
n_attack_split = 10
X_attack, Y_attack, plt_attack, index = load_attack_dataset_sss(dataset_path, n_attack_split, attack_limit=10000, 
                                                        scalers_list=scalers_list, samples=0)
print (X_attack[0:10])
print (Y_attack[0:10])
print (plt_attack[0:10])
print (index[0:10])


print ('===============================load_dataset_sss')
scalers_list = [StandardScaler(), MinMaxScaler()]
n_train_split = 30
profile_limit = 60000
X_profiling, Y_profiling, index = load_dataset_sss(dataset_path, n_train_split, train_limit=profile_limit, scalers_list=scalers_list)
print (X_profiling[0:10])
print (Y_profiling[0:10])
print (index[0:10])


print ('===============================load_dataset_sss(train, val)')
scalers_list = [StandardScaler(), MinMaxScaler()]
n_train_split = 30
train_limit = 60000
n_val_split = 5
val_limit = 70000
(X_profiling, Y_profiling, index), (X_val, Y_val, index_val) = load_dataset_sss(dataset_path, n_train_split, train_limit=train_limit, 
                                                    scalers_list=scalers_list, n_val_split=n_val_split, val_limit=val_limit)
print (X_profiling[0:10])
print (Y_profiling[0:10])
print (index[0:10])

print ('---------------------')
print (X_val[0:10])
print (Y_val[0:10])
print (index_val[0:10])


print ('===============================load_dataset_sss_random')
scalers_list = [StandardScaler(), MinMaxScaler()]
X_profiling, Y_profiling, index = load_dataset_sss_random(dataset_path, n_train_split, train_limit=profile_limit, scalers_list=scalers_list)

print (X_profiling[0:10])
print (Y_profiling[0:10])
print (index[0:10])

print ('===============================load_dataset_sss_random (train, val)')

scalers_list = [StandardScaler(), MinMaxScaler()]
(X_profiling, Y_profiling, index), (X_val, Y_val, index_val) = load_dataset_sss_random(dataset_path, n_train_split, train_limit=profile_limit, 
                                                          scalers_list=scalers_list, n_val_split=n_val_split, val_limit=val_limit)

print (X_profiling[0:10])
print (Y_profiling[0:10])
print (index[0:10])

print ('---------------------')
print (X_val[0:10])
print (Y_val[0:10])
print (index_val[0:10])
