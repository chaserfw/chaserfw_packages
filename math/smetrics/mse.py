from sfileengine import FileEngine
from sutils import tqdm
import numpy as np

def MSE(original_fe:FileEngine, predicted_fe:FileEngine, ntraces=None, batch_size=256):
	ntraces            = original_fe.TotalTraces if ntraces is None else ntraces
	n_samples          = original_fe.TotalSamples * ntraces
	partial_division   = 0
	batches_index      = 0
	# Fixed number of batches for bigger file engine
	nbatches = int(ntraces / batch_size) if (ntraces % batch_size) == 0 else int((ntraces - (ntraces % batch_size)) / batch_size)
	pbar1 = tqdm(total=ntraces, desc='[INFO]: Computing MSE', position=0)
	done = False
	while not (done):
		if batches_index <= nbatches:
			# Set the next trace
			lower_limit = batches_index * batch_size
			upper_limit = (batches_index + 1) * batch_size
			if (batches_index == nbatches):
				upper_limit = (batches_index * batch_size) + (ntraces % batch_size)
				
			original_trace_meta = np.array(original_fe[lower_limit:upper_limit], dtype=np.ndarray)
			predicted_trace_meta = np.array(predicted_fe[lower_limit:upper_limit], dtype=np.ndarray)
			
			original_traces = np.vstack(original_trace_meta[:,0])
			predicted_traces = np.vstack(predicted_trace_meta[:,0])
			
			cumulative = np.sum((original_traces-predicted_traces)**2)
			partial_division = cumulative + partial_division
			
			if batches_index == nbatches:
				pbar1.update(ntraces % batch_size)
				pbar1.close()
			else:
				pbar1.update(batch_size)
			batches_index = batches_index + 1
		else:
			done = True
	return np.true_divide(partial_division, n_samples)