from sfileengine import FileEngine
from builders import ASCADBuilder
from sutils import tqdm
import numpy as np
import os
#===========================================================================================
def __one_file_scaling(ascadBuilder:ASCADBuilder, pro_fe:FileEngine, att_fe:FileEngine, scalers_list, 
						pro_fe_ntraces=None, batch_size=256):
	pro_batches_index = 0	
	if pro_fe_ntraces is None:
		pro_fe_ntraces = pro_fe.TotalTraces
	
	# Fixed number of batches for bigger file engine
	pro_fe_nbatches = int(pro_fe_ntraces / batch_size) if (pro_fe_ntraces % batch_size) == 0 else int((pro_fe_ntraces - (pro_fe_ntraces % batch_size)) / batch_size)

	pbar1 = tqdm(total=pro_fe_ntraces, position=0, desc='[INFO *BatchingReverseScaling*]: Reverse scaling')
	pro_done = False
	while not (pro_done):		
		if pro_batches_index <= pro_fe_nbatches:
			# Set the next trace
			lower_limit = pro_batches_index * batch_size
			upper_limit = (pro_batches_index + 1) * batch_size
			if (pro_batches_index == pro_fe_nbatches):
				upper_limit = (pro_batches_index * batch_size) + (pro_fe_ntraces % batch_size)

			trace_meta = np.array(pro_fe[lower_limit:upper_limit], dtype=np.ndarray)

			# Get the trace from the smaller file engine
			traces = np.vstack(trace_meta[:,0])
			# Get metadata vector 
			tuplas = np.vstack(trace_meta[:,1])
			# Get the label of the trace
			labels = trace_meta[:,2]
		
			for scaler in scalers_list:
				traces = scaler.inverse_transform(traces)

			if (len(traces.shape) == 3):
				traces = np.squeeze(traces)

			ascadBuilder.Feed_Batch_Traces(batch_size, ptraces=traces, pmetadata=tuplas, labeler=labels, flush_flag=1000)
			if pro_batches_index == pro_fe_nbatches:
				pbar1.update(pro_fe_ntraces % batch_size)
				pbar1.close()
			else:
				pbar1.update(batch_size)
			# Update batches index of the bigger fileengine
			pro_batches_index = pro_batches_index + 1
		else:
			pro_done = True
	print ()
#===========================================================================================
def __two_files_scaling(ascadBuilder:ASCADBuilder, pro_fe:FileEngine, att_fe:FileEngine, scalers_list, pro_fe_ntraces=None, 
						att_fe_ntraces=None, batch_size=256):
	pro_batches_index = 0
	att_batches_index = 0
	
	pro_fe_ntraces = pro_fe.TotalTraces if pro_fe_ntraces is None or pro_fe.TotalTraces < pro_fe_ntraces else pro_fe_ntraces
	att_fe_ntraces = att_fe.TotalTraces if att_fe_ntraces is None or att_fe.TotalTraces < att_fe_ntraces else att_fe_ntraces
	
	# Fixed number of batches for bigger file engine
	pro_fe_nbatches = int(pro_fe_ntraces / batch_size) if (pro_fe_ntraces % batch_size) == 0 else int((pro_fe_ntraces - (pro_fe_ntraces % batch_size)) / batch_size)
	# Fixed number of batches for smaller file engine
	att_fe_nbatches = int(att_fe_ntraces / batch_size) if (att_fe_ntraces % batch_size) == 0 else int((att_fe_ntraces - (att_fe_ntraces % batch_size)) / batch_size)
	
	pbar1 = tqdm(total=pro_fe_ntraces, position=0, desc='[INFO *BatchingReverseScaling*]: FE1 Reverse scaling')
	pbar2 = tqdm(total=att_fe_ntraces, position=1, desc='[INFO *BatchingReverseScaling*]: FE2 Reverse scaling')
	pro_done = att_done = False
	while not (pro_done and att_done):
		# check if the smaller file engine is done
		if att_batches_index <= att_fe_nbatches:
			# Set the next trace
			lower_limit = att_batches_index * batch_size
			upper_limit = (att_batches_index + 1) * batch_size
			if (att_batches_index == att_fe_nbatches):
				upper_limit = (att_batches_index * batch_size) + (att_fe_ntraces % batch_size)
			trace_meta = np.array(att_fe[lower_limit:upper_limit], dtype=np.ndarray)
			
			# Get the trace from the smaller file engine
			traces = np.vstack(trace_meta[:,0])
			# Get metadata vector 
			tuplas = np.vstack(trace_meta[:,1])
			# Get the label of the trace
			labels = trace_meta[:,2]
		
			for scaler in scalers_list:
				traces = scaler.inverse_transform(traces)			
			if (len(traces.shape) == 3):
				traces = np.squeeze(traces)

			ascadBuilder.Feed_Batch_Traces(batch_size, atraces=traces, ametadata=tuplas, labeler=labels, flush_flag=1000)
			
			if att_batches_index == att_fe_nbatches:
				pbar2.update(att_fe_ntraces % batch_size)
				pbar2.close()
			else:
				pbar2.update(batch_size)
			# Update batches index of the smaller fileengine
			att_batches_index = att_batches_index + 1
		else:
			att_done = True
		
		if pro_batches_index <= pro_fe_nbatches:
			# Set the next trace
			pro_lower_limit = pro_batches_index * batch_size
			pro_upper_limit = (pro_batches_index + 1) * batch_size
			if (pro_batches_index == pro_fe_nbatches):
				pro_upper_limit = (pro_batches_index * batch_size) + (pro_fe_ntraces % batch_size)
			trace_meta = np.array(pro_fe[pro_lower_limit:pro_upper_limit], dtype=np.ndarray)

			# Get the trace from the smaller file engine
			traces = np.vstack(trace_meta[:,0])
			# Get metadata vector 
			tuplas = np.vstack(trace_meta[:,1])
			# Get the label of the trace
			labels = trace_meta[:,2]
		
			for scaler in scalers_list:
				traces = scaler.inverse_transform(traces)

			if (len(traces.shape) == 3):
				traces = np.squeeze(traces)

			ascadBuilder.Feed_Batch_Traces(batch_size, ptraces=traces, pmetadata=tuplas, labeler=labels, flush_flag=1000)

			if pro_batches_index == pro_fe_nbatches:
				pbar1.update(pro_fe_ntraces % batch_size)
				pbar1.close()
			else:
				pbar1.update(batch_size)

			# Update batches index of the bigger fileengine
			pro_batches_index = pro_batches_index + 1
		else:
			pro_done = True
	print ()
#===========================================================================================
def batching_reverse_scaling(pro_fe:FileEngine, scalers_list:list, att_fe:FileEngine=None, pro_fe_ntraces:int=None, 
					 att_fe_ntraces:int=None, batch_size:int=256, dest_dir:str=None, nsamples:int=None, 
					 mtd_descriptor=None, **kwargs):	
	# Create destination path
	dataset_path = pro_fe.Path
	if dest_dir is not None:
		dataset_path = dest_dir

	if nsamples is None:
		nsamples = pro_fe.TotalSamples

	if mtd_descriptor is None:
		mtd_descriptor = pro_fe.MetadataTypeDescriptor
	
	suffix = 'scaled'
	if 'suffix' in kwargs:
		suffix = kwargs['suffix']
	
	filename = pro_fe.FileName

	output_path = os.path.join(dataset_path, '{}-{}.h5'.format(filename, suffix))
	print ('[INFO *BatchingReverseScaling*]: Output path:', output_path)
	
	# create the ASCAD builder
	ascadBuilder = ASCADBuilder(output_path, descriptor=mtd_descriptor)
	ascadBuilder.Set_Profiling(nsamples)	
	ascadBuilder.Add_Profiling_Label()
	if att_fe is not None:
		ascadBuilder.Set_Attack(nsamples)
		ascadBuilder.Add_Attack_Label()
		__two_files_scaling(ascadBuilder, pro_fe, att_fe, scalers_list, pro_fe_ntraces=pro_fe_ntraces, 
							att_fe_ntraces=att_fe_ntraces, batch_size=batch_size)	
	else:
		__one_file_scaling(ascadBuilder, pro_fe, scalers_list, pro_fe_ntraces=pro_fe_ntraces, 
							batch_size=batch_size)
	ascadBuilder.close()