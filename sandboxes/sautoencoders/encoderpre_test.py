from sautoencoder import EncoderPre
from builders import MetadataTypeDescriptor
from sfileengine import H5FileEngine
import tensorflow as tf
import pickle
import os
import numpy as np

def merging(data):
    merged = np.concatenate(list(data), axis=0)
    return merged.astype(np.uint8)

id_folder = '20210704-173202'
train_result_path = r'.\test_model\20210704-173202'.format(id_folder)
model_path = os.path.join(train_result_path, 'encoder.h5')
dataset_path = r'D:\SCA\datasets\ASCAD_var_desync100.h5'
encoder_n_samples = 700

profiling_file_engine = H5FileEngine(dataset_path, group='/Profiling_traces')
attack_file_engine = H5FileEngine(dataset_path, group='/Attack_traces')

print (profiling_file_engine.MetadataTypeDescriptor)
print (type(profiling_file_engine.MetadataTypeDescriptor))

scalers_list    = [os.path.join(train_result_path, 'scaler_0_{}'.format('autoencoder')), 
                   os.path.join(train_result_path, 'scaler_1_{}'.format('autoencoder'))]
storaged_scaler = []
for scaler in scalers_list:
	with open(scaler, 'rb') as f:
		storaged_scaler.append(pickle.load(f))

# Not used because the descriptor is inherited from the H5FileEngine
mtd = MetadataTypeDescriptor()
mtd.Add_Descriptor('plaintext', np.uint8, 16)
mtd.Add_Descriptor('key', np.uint8, 16)
mtd.Add_Descriptor('masks', np.uint8, 18)
#mtd.Add_Descriptor('ciphertext', np.uint8, 16)
mtd.Add_Descriptor('fixed', np.uint32, 1)


model = tf.keras.models.load_model(model_path, compile=False)
encoderPre = EncoderPre(profiling_file_engine, model, encoder_n_samples, add_id=id_folder, des_scalers_list=storaged_scaler, 
						second_file_engine=attack_file_engine, fengine_total_traces=2000, 
						second_fengine_total_traces=1000, batch_size=256)

encoderPre.process()
profiling_file_engine.close()
attack_file_engine.close()