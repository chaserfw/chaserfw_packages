from sfileengine import h5_group_descriptor
from builders import ASCADBuilder
from builders import MetadataTypeDescriptor
import tables
import numpy as np
from tqdm import trange
from scryptoutils import AES_Sbox
import os

target_board = 'D1'
derivation   = 'masked'
dataset_path = r'D:AES_PT'
h5 = tables.open_file(os.path.join(dataset_path, 'AES_PT.h5'))
target_group = h5.root.D1.Masked
n_samples = 1225 if derivation != 'masked' else 2300
n_random_key_traces = 100000 if derivation != 'masked' else 200000
n_fixed_key_traces  = 50000 if derivation != 'masked' else 100000

print (target_group.RndmKey.Data.Plaintext[14])
print (target_group.RndmKey.Data.Key[14])
print (target_group.RndmKey.Data.Ciphertext[14])
print ('----')
print (target_group.RndmKey.Data.Plaintext[15])
print (target_group.RndmKey.Data.Key[15])
print (target_group.RndmKey.Data.Ciphertext[15])
print ('----')
print (target_group.RndmKey.Data.Plaintext[16])
print (target_group.RndmKey.Data.Key[16])
print (target_group.RndmKey.Data.Ciphertext[16])
print ('----')
print (target_group.RndmKey.Data.Plaintext[17])
print (target_group.RndmKey.Data.Key[17])
print (target_group.RndmKey.Data.Ciphertext[17])
print ('----')


mtd = MetadataTypeDescriptor()
mtd.Add_Descriptor('plaintext', np.uint8, 16)
mtd.Add_Descriptor('key', np.uint8, 16)
if derivation == 'masked':
    mtd.Add_Descriptor('masks', np.uint8, 2)
mtd.Add_Descriptor('ciphertext', np.uint8, 16)
mtd.Add_Descriptor('fixed', np.uint8, 1)

ascadBuilder = ASCADBuilder(os.path.join(dataset_path, 'AES_PT_{}-{}-label-0.h5'.format(derivation, target_board)), descriptor=mtd)
ascadBuilder.Set_Profiling(n_samples)
ascadBuilder.Set_Attack(n_samples)
ascadBuilder.Add_Attack_Label()
ascadBuilder.Add_Profiling_Label()


for i in trange(n_random_key_traces, desc='[INFO]: Creating {}-{} dataset'.format(derivation, target_board)):
#for i in trange(100, desc='[INFO]: Creating {}-{} dataset'.format(derivation, target_board)):
    p = target_group.RndmKey.Data.Plaintext[i]
    k = target_group.RndmKey.Data.Key[i]
    c = target_group.RndmKey.Data.Ciphertext[i]
    if derivation == 'masked':
        m = target_group.RndmKey.Data.Masks[i]
    
    if i < n_fixed_key_traces:
        f_p = target_group.FixedKey.Data.Plaintext[i]
        f_k = target_group.FixedKey.Data.Key[i]
        f_c = target_group.FixedKey.Data.Ciphertext[i]
        f_m = target_group.FixedKey.Data.Masks[i]
        if not (f_p.shape[0] <= 15 or f_k.shape[0] <= 15 or f_c.shape[0] <= 15):
            fixed_meta = np.array([(f_p, f_k, f_c, [1])],  dtype=mtd.Descriptor) if derivation != 'masked' else np.array([(f_p, f_k, f_m, f_c, [1])],  dtype=mtd.Descriptor)
            ascadBuilder.Feed_Traces(atraces=[target_group.FixedKey.Traces[i]], ametadata=fixed_meta, 
            labeler=[AES_Sbox[f_p[0] ^ f_k[0]]])
        else:
            print ('wrong in fixed:', i)

    if not (p.shape[0] <= 15 or k.shape[0] <= 15 or c.shape[0] <= 15):
        random_meta = np.array([(p, k, c, [0])], dtype=mtd.Descriptor) if derivation != 'masked' else np.array([(p, k, m, c, [0])], dtype=mtd.Descriptor)
        ascadBuilder.Feed_Traces(ptraces=[target_group.RndmKey.Traces[i]], pmetadata=random_meta, 
        labeler=[AES_Sbox[p[0] ^ k[0]]])
    else:
        print ('wrong in random:', i)

ascadBuilder.Close_File()

h5.close()
