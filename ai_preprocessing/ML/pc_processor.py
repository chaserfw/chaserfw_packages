"""
@author: Servio Paguada
@email: serviopaguada@gmail.com
"""
import os
import numpy as np
from builders import ASCADBuilder
from sfileengine import H5FileEngine
from sutils import tqdm

class PCProcessor:
	def __init__(self, pro_fe:H5FileEngine, pc_model, n_samples:int=None,
				dest_dir:str=None, scalers_list:list=None, mtd_descriptor=None, 
				att_fe:H5FileEngine=None, batch_size:int=256, pro_fe_ntraces:int=None, 
				att_fe_ntraces:int=None, **kwarg):
		"""Class provides an interface to process traces using an extractor model or its encoder part, 
			such a model should perform prediction by default, meaning, the class will
			use de function tensorflow.keras.models.Model.predict direclty from the given model.
			While and after processing an ASCAD file will be built as a result.
			
		:param pro_fe: H5FileEngine that maps to the profiling traces.
		:type pro_fe: :class:`H5FileEngine`

		:param pc_model: Principal Component model, it could be any of the principal component base 
			models (PCA, LDA, Kernel PCA, KDA) that implements a `transform` function.

		:param n_samples: The number of samples (features) of the traces in both profiling and 
			attack group.

		:param dest_dir: The path where the resultant file will be written.

		:param scalers_list: List of scalers used to train ther PC model.

		:param mtd_descriptor: Descriptor that the ASCADBuilder uses to build the metadata type.

		:param att_fe: H5FileEngine that maps to the attack traces

		:param batch_size: The batch size of the whole set of traces used to iterate through it.
		:type batch_size: `int`

		:param pro_fe_ntraces: The upper limit of number of traces taken from the original H5FileEngine
			profiling traces, and used to build the resultant file.

		:param att_fe_ntraces: The upper limit of number of traces taken from the original H5FileEngine
			attack traces, and used to build the resultant file.

		:param **kwarg: 
			suffix (str): the suffix that will be concatenated to the file name
		"""	
		super().__init__()
		self.__suffix = 'compressed'
		if 'suffix' in kwarg:
			self.__suffix = kwarg['suffix']

		if mtd_descriptor is None:
			self.__mtype_descriptor = pro_fe.MetadataTypeDescriptor
		else:
			self.__mtype_descriptor = mtd_descriptor.Descriptor

		self.__scalers_list = scalers_list
		self.__pc_model = pc_model
		
		if n_samples is None:
			n_samples = pro_fe.TotalSamples

		self.__dest_dir = None
		if dest_dir is not None:
			self.__dest_dir = dest_dir

		self.__pro_fe         = pro_fe
		self.__att_fe         = att_fe
		self.__n_samples      = n_samples
		self.__ascadBuilder   = None
		self.__batch_size     = batch_size
		self.__pro_fe_ntraces = pro_fe_ntraces
		self.__att_fe_ntraces = att_fe_ntraces
		
		if att_fe is not None:
			self.__process_function = self.__two_file_engine_function_batches
		else:
			self.__process_function = self.__one_file_engine_function

	def process(self):
		# defining two files, Autoencoder and encoder traces version
		
		# Create destination path
		dataset_path = self.__pro_fe.Path
		if self.__dest_dir is not None:
			dataset_path = self.__dest_dir
		
		filename = self.__pro_fe.FileName

		output_path = os.path.join(dataset_path, '{}-{}.h5'.format(filename, self.__suffix))
		print ('[INFO *PCProcessor*] Output path:', output_path)
		
		# create the ASCAD builder
		self.__ascadBuilder = ASCADBuilder(output_path, descriptor=self.__mtype_descriptor)
		self.__ascadBuilder.Set_Profiling(self.__n_samples)
		self.__ascadBuilder.Set_Attack(self.__n_samples)
		self.__ascadBuilder.Add_Attack_Label()
		self.__ascadBuilder.Add_Profiling_Label()
		self.__process_function()
		self.close_created_file()

	def __one_file_engine_function(self):
		pass

	def __two_file_engine_function_batches(self):

		pro_batches_index = 0
		att_batches_index = 0
	
		self.__pro_fe_ntraces = self.__pro_fe.TotalTraces if self.__pro_fe_ntraces is None or self.__pro_fe.TotalTraces < self.__pro_fe_ntraces else self.__pro_fe_ntraces
		self.__att_fe_ntraces = self.__att_fe.TotalTraces if self.__att_fe_ntraces is None or self.__att_fe.TotalTraces < self.__att_fe_ntraces else self.__att_fe_ntraces
		
		# Fixed number of batches for bigger file engine
		pro_fe_nbatches = int(self.__pro_fe_ntraces / self.__batch_size) if (self.__pro_fe_ntraces % self.__batch_size) == 0 else int((self.__pro_fe_ntraces - (self.__pro_fe_ntraces % self.__batch_size)) / self.__batch_size)
		# Fixed number of batches for smaller file engine
		att_fe_nbatches = int(self.__att_fe_ntraces / self.__batch_size) if (self.__att_fe_ntraces % self.__batch_size) == 0 else int((self.__att_fe_ntraces - (self.__att_fe_ntraces % self.__batch_size)) / self.__batch_size)

		pbar1 = tqdm(total=self.__pro_fe_ntraces, position=0)
		pbar2 = tqdm(total=self.__att_fe_ntraces, position=1)
		pro_done = att_done = False
		while not (pro_done and att_done):
			# check if the smaller file engine is done
			if att_batches_index <= att_fe_nbatches:

				# Set the next trace
				lower_limit = att_batches_index * self.__batch_size
				upper_limit = (att_batches_index + 1) * self.__batch_size
				if (att_batches_index == att_fe_nbatches):
					upper_limit = (att_batches_index * self.__batch_size) + (self.__att_fe_ntraces % self.__batch_size)
				# Set the next trace
				trace_meta = np.array(self.__att_fe[lower_limit:upper_limit], dtype=np.ndarray)
				# Get the trace from the smaller file engine
				traces = np.vstack(trace_meta[:,0])
				# Get metadata vector 
				tuplas = np.vstack(trace_meta[:,1])
				# Get the label of the trace
				labels = trace_meta[:,2]

				for scaler in self.__scalers_list:
					traces = scaler.transform(traces)

				# Reshaping traces
				transformed_traces = self.__pc_model.transform(traces)

				self.__ascadBuilder.Feed_Batch_Traces(self.__batch_size, atraces=transformed_traces, ametadata=tuplas, labeler=labels, flush_flag=1000)
				
				if att_batches_index == att_fe_nbatches:
					pbar2.update(self.__att_fe_ntraces % self.__batch_size)
					pbar2.close()
				else:
					pbar2.update(self.__batch_size)
				# Update batches index of the smaller fileengine
				att_batches_index = att_batches_index + 1
			else:
				att_done = True
			
			if pro_batches_index <= pro_fe_nbatches:
				# Set the next trace
				# Set the next trace
				pro_lower_limit = pro_batches_index * self.__batch_size
				pro_upper_limit = (pro_batches_index + 1) * self.__batch_size
				if (pro_batches_index == pro_fe_nbatches):
					pro_upper_limit = (pro_batches_index * self.__batch_size) + (self.__pro_fe_ntraces % self.__batch_size)

				trace_meta = np.array(self.__pro_fe[pro_lower_limit:pro_upper_limit], dtype=np.ndarray)
				# Get the trace from the smaller file engine
				traces = np.vstack(trace_meta[:,0])
				# Get metadata vector 
				tuplas = np.vstack(trace_meta[:,1])
				# Get the label of the trace
				labels = trace_meta[:,2]
				for scaler in self.__scalers_list:
					traces = scaler.transform(traces)
				
				transformed_traces = self.__pc_model.transform(traces)

				self.__ascadBuilder.Feed_Batch_Traces(self.__batch_size, ptraces=transformed_traces, pmetadata=tuplas, labeler=labels, flush_flag=1000)

				if pro_batches_index == pro_fe_nbatches:
					pbar1.update(self.__pro_fe_ntraces % self.__batch_size)
					pbar1.close()
				else:
					pbar1.update(self.__batch_size)
				# Update batches index of the bigger fileengine
				pro_batches_index = pro_batches_index + 1
			else:
				pro_done = True
	
		print ()

	def __del__(self):
		# closing        
		if self.__ascadBuilder is not None:
			self.__ascadBuilder.Close_File()
	
	def close_created_file(self):
		if self.__ascadBuilder is not None:
			self.__ascadBuilder.close()