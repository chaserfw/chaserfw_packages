from sfileengine import h5_group_descriptor
from builders import ASCADBuilder
from builders import MetadataTypeDescriptor
import tables
import numpy as np
from tqdm import trange
from scryptoutils import AES_Sbox
import os

model_id = '20200430-120909'
dataset_path = r'D:\cw_datasets'

path = os.path.join(dataset_path, model_id, 'ascad_acq_{}.h5'.format(model_id))
h5 = tables.open_file(path)

mtd = MetadataTypeDescriptor()
mtd.Add_Descriptor('plaintext', np.uint8, 16)
mtd.Add_Descriptor('key', np.uint8, 16)
mtd.Add_Descriptor('ciphertext', np.uint8, 16)
mtd.Add_Descriptor('fixed', np.uint8, (1,))

#n_samples = h5.root.traces.shape[1]
n_samples = 1800
ascadBuilder = ASCADBuilder(os.path.join(dataset_path, model_id, '{}-sep-1800.h5'.format(model_id)), descriptor=mtd)
ascadBuilder.Set_Profiling(n_samples)
ascadBuilder.Set_Attack(n_samples)
ascadBuilder.Add_Attack_Label()
ascadBuilder.Add_Profiling_Label()

for i in trange(h5.root.traces.shape[0], desc='[INFO]: Creating {} dataset'.format(model_id)):
#for i in trange(100):
    meta = np.array([(h5.root.metadata[i]['plaintext'],
                        h5.root.metadata[i]['key'],
                        h5.root.metadata[i]['ciphertext'],
                        h5.root.metadata[i]['fixed'])], 
                        dtype=mtd.Descriptor)
    if h5.root.metadata[i]['fixed'] == 1:
        ascadBuilder.Feed_Traces(atraces=[h5.root.traces[i][1200:3000]], ametadata=meta, 
        labeler=[AES_Sbox[h5.root.metadata[i]['plaintext'][2] ^ h5.root.metadata[i]['key'][2]]])

    else:
        ascadBuilder.Feed_Traces(ptraces=[h5.root.traces[i][1200:3000]], pmetadata=meta,
        labeler=[AES_Sbox[h5.root.metadata[i]['plaintext'][2] ^ h5.root.metadata[i]['key'][2]]])

ascadBuilder.Close_File()
h5.close()