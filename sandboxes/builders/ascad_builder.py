from builders import ASCADBuilder
from builders import MetadataTypeDescriptor
import trsfile
from tqdm import trange 
import numpy as np

trs_file = trsfile.open(r'C:\Users\slpaguada\ChipWhisperer5_64\chipwhisperer\jupyter\Servio\traces_storaged\acq_20201008-141959\acq_20201008-141959.trs')
n_traces = len(trs_file)
n_samples = len(trs_file[0])

mdataD = MetadataTypeDescriptor()
mdataD.Add_Descriptor('plaintext')
mdataD.Add_Descriptor('key')
mdataD.Add_Descriptor('ciphertext')

ascadBuilder = ASCADBuilder('./ascad_test.h5', mdataD)
ascadBuilder.Set_Profiling(n_samples)
#TODO: la metadata debe tener esta forma: np.array([(trace.textin, trace.key, trace.textout, mask, fr_index)]
#		la traza debe tener esta forma: [np.array(trace.wave, dtype='float32')]

for i in trange(n_traces, desc='[INFO]: Converting'):
    trace = [np.array(trs_file[i], dtype='float32')]

    textin = np.frombuffer(trs_file[i].data[0:16], dtype=np.uint8)    
    textout = np.frombuffer(trs_file[i].data[0:16], dtype=np.uint8)    
    key = np.frombuffer(trs_file[i].data[0:16], dtype=np.uint8)    
    
    metadata = np.array([(textin, key, textout)], dtype=mdataD.Descriptor)
    ascadBuilder.Feed_Traces(ptraces=trace, pmetadata=metadata)
    
ascadBuilder.Close_File()
trs_file.close()