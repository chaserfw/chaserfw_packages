from builders import ASCADBuilder
from builders import MetadataTypeDescriptor
import tables
import numpy as np
from tqdm import trange
from scryptoutils import AES_Sbox
import os

#model_id = '20200430-120909'
model_id = '20210108-114638'
dataset_path = 'D:\\'

path = os.path.join(dataset_path, 'ASCAD_EPF_2_{}.h5'.format(model_id))
h5 = tables.open_file(path)

mtd = MetadataTypeDescriptor()
mtd.Add_Descriptor('plaintext', np.uint8, 16)
mtd.Add_Descriptor('key', np.uint8, 16)
mtd.Add_Descriptor('masks', np.uint8, 18)
mtd.Add_Descriptor('desync', np.uint8, 1)

#n_samples = h5.root.traces.shape[1]
n_samples = 560
ascadBuilder = ASCADBuilder(os.path.join(dataset_path, 'ASCAD_EPF_{}.h5'.format(model_id)), descriptor=mtd)
ascadBuilder.Set_Profiling(n_samples)
ascadBuilder.Set_Attack(n_samples)
ascadBuilder.Add_Attack_Label()
ascadBuilder.Add_Profiling_Label()

for i in trange(h5.root.Profiling_traces.traces.shape[0], desc='[INFO]: Creating {} dataset'.format(model_id)):
    
    if i < h5.root.Attack_traces.traces.nrows:
        meta = np.array([(h5.root.Attack_traces.metadata[i]['plaintext'],
                            h5.root.Attack_traces.metadata[i]['key'],
                            h5.root.Attack_traces.metadata[i]['masks'],
                            h5.root.Attack_traces.metadata[i]['desync'])], 
                            dtype=mtd.Descriptor)

        ascadBuilder.Feed_Traces(atraces=[h5.root.Attack_traces.traces[i]], ametadata=meta, labeler=[h5.root.Attack_traces.labels[i]])

    
    meta = np.array([(h5.root.Profiling_traces.metadata[i]['plaintext'],
                            h5.root.Profiling_traces.metadata[i]['key'],
                            h5.root.Profiling_traces.metadata[i]['masks'],
                            h5.root.Profiling_traces.metadata[i]['desync'])], 
                            dtype=mtd.Descriptor)        
    ascadBuilder.Feed_Traces(ptraces=[h5.root.Profiling_traces.traces[i]], pmetadata=meta, labeler=[h5.root.Profiling_traces.labels[i]])

ascadBuilder.Close_File()
h5.close()