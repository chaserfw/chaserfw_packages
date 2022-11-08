from processors import batching_reverse_scaling
from sutils import load_scaler
from sfileengine import H5FileEngine
import os

path = r'..\..\sautoencoders\test_model\20210704-173202'

storaged_scalers = [os.path.join(path, 'scaler_0_autoencoder'), 
                    os.path.join(path, 'scaler_1_autoencoder')]

scalers_list = []
for scaler in storaged_scalers:
    scalers_list.append(load_scaler(scaler))

dataset_path = os.path.join(path, 'ASCAD_var_desync100-whole-clean.h5')
pro_fe = H5FileEngine(dataset_path, 'Profiling_traces')
att_fe = H5FileEngine(dataset_path, 'Attack_traces')

print (pro_fe.TotalSamples)

scalers_list.reverse()
batching_reverse_scaling(pro_fe, scalers_list, att_fe, 2000, 2000, dest_dir='.')
scalers_list.reverse()
pro_fe.close()
att_fe.close()


print ('[INFO]: Reading resultant file')
resultant_pro_fe = H5FileEngine('ASCAD_var_desync100-whole-clean-scaled.h5', group='/Profiling_traces')
resultant_att_fe = H5FileEngine('ASCAD_var_desync100-whole-clean-scaled.h5', group='/Attack_traces')

print (resultant_pro_fe[0])
print (resultant_att_fe[0])

resultant_pro_fe.close()
resultant_att_fe.close()