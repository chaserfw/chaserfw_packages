"""
@author: Servio Paguada
@email: serviopaguada@gmail.com

This examples uses:
    Classes: AutoEncoderProcessor, H5FileEngine, MetadataTypeDescriptor (deactivated)
    Functions: load_scaler
"""
from sautoencoder import AutoEncoderProcessor
#from builders import MetadataTypeDescriptor
from sfileengine import H5FileEngine
import tensorflow as tf
from sutils import load_scaler
import os
import numpy as np

# Deactivated since the metadata type descriptor used is inherited from the H5FileEngine
def merging(data):
    merged = np.concatenate(list(data), axis=0)
    return merged.astype(np.uint8)

# Targeting training dir
id_folder = '20210704-173202'
train_dir = r'.\test_model\{}'.format(id_folder)

# When using encoder the nsamples is equal to the code dimension,
# while using autoencoder (the whole model) the nsamples is equal to the output ot the autoencoder
# or the dimention of the original signals.
model_path = os.path.join(train_dir, 'encoder.h5')
dataset_path = r'D:\SCA\datasets\ASCAD_var_desync100.h5'

pro_fe = H5FileEngine(dataset_path, group='/Profiling_traces')
att_fe = H5FileEngine(dataset_path, group='/Attack_traces')

scalers_list    = [os.path.join(train_dir, 'scaler_0_{}'.format('autoencoder')), 
                   os.path.join(train_dir, 'scaler_1_{}'.format('autoencoder'))]
storaged_scaler = []
for scaler in scalers_list:
	storaged_scaler.append(load_scaler(scaler))

# Not used because the descriptor is inherited from the H5FileEngine
# mtd = MetadataTypeDescriptor()
# mtd.Add_Descriptor('plaintext', np.uint8, 16)
# mtd.Add_Descriptor('key', np.uint8, 16)
# mtd.Add_Descriptor('masks', np.uint8, 2)
# mtd.Add_Descriptor('ciphertext', np.uint8, 16)
# mtd.Add_Descriptor('fixed', np.uint8, 1)

model = tf.keras.models.load_model(model_path, compile=False)
autoencoderPre = AutoEncoderProcessor(pro_fe, model, scalers_list=storaged_scaler, 
						att_fe=att_fe, pro_fe_ntraces=2000, 
						att_fe_ntraces=1000, batch_size=256, 
                        dest_dir=train_dir, suffix='compress', n_samples=700)

autoencoderPre.process()
pro_fe.close()
att_fe.close()