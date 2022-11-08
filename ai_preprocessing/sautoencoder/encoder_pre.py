import os
import tables
import numpy as np
from enum import Enum
from svortex import ArrayVortex
from svortex import FormatConfig
from builders import ASCADBuilder
from builders import MetadataTypeDescriptor
import sys
from sutils import tqdm


class EncoderPre:
	def __init__(self, file_engine, encoder_model, encoder_n_samples, add_id=None, 
				 training_dir=None, des_scalers_list=None, mtd_descriptor=None, 
				 second_file_engine=None, batch_size=None, fengine_total_traces=None, 
				 second_fengine_total_traces=None, **kwarg):
		'''kwarg arguments fdataset_path
		'''
		super().__init__()

		if mtd_descriptor is None:
			self.__mtype_descripter = file_engine.MetadataTypeDescriptor
		else:
			self.__mtype_descripter = mtd_descriptor.Descriptor

		if training_dir is None:
			self.__des_scalers_list = des_scalers_list
			self.__encoder_model = encoder_model
		else:
			print ('[INFO]: training dir provided, attempting to gather training info from json file...')

		self.__feature_dataset_path_destination = None
		if 'fdataset_path' in kwarg:
			self.__feature_dataset_path_destination = kwarg['fdataset_path']

		self.__file_engine        = file_engine
		self.__second_file_engine = second_file_engine
		self.__encoder_n_samples  = encoder_n_samples
		self.__ascadBuilder       = None
		self.__batch_size         = batch_size
		self.__fengine_total_traces = fengine_total_traces
		self.__second_fengine_total_traces = second_fengine_total_traces
		self.__add_id = add_id
		
		if second_file_engine is not None:
			if self.__batch_size is None:
				self.__process_function = self.__two_file_engine_function
			else:
				self.__process_function = self.__two_file_engine_function_batches
		else:
			self.__process_function = self.__one_file_engine_function

	def process(self):
		# defining two files, Autoencoder and encoder traces version
		
		# Create destination path
		dataset_path = self.__file_engine.Path
		if self.__feature_dataset_path_destination is not None:
			dataset_path = self.__feature_dataset_path_destination
		
		filename     = self.__file_engine.FileName

		print ('*-------')
		print (os.path.join(dataset_path, '{}-{}.h5'.format(filename, 'compressed')))
		print ('*-------')
		# create the ASCAD builder
		self.__ascadBuilder = ASCADBuilder(os.path.join(dataset_path, '{}-{}-{}.h5'.format(filename, 'compressed', self.__add_id)), descriptor=self.__mtype_descripter)
		self.__ascadBuilder.Set_Profiling(self.__encoder_n_samples)
		self.__ascadBuilder.Set_Attack(self.__encoder_n_samples)
		self.__ascadBuilder.Add_Attack_Label()
		self.__ascadBuilder.Add_Profiling_Label()
		self.__process_function()
		self.close_created_file()

	def __one_file_engine_function(self):
		pass

	def __two_file_engine_function(self):

		trace_counter = 0
		smaller_file_engine = self.__file_engine if self.__file_engine.TotalTraces < self.__second_file_engine.TotalTraces else self.__second_file_engine
		bigger_file_engine = self.__file_engine if self.__file_engine.TotalTraces > self.__second_file_engine.TotalTraces else self.__second_file_engine

		pbar1 = tqdm(total=self.__file_engine.TotalTraces, position=0)
		pbar2 = tqdm(total=self.__second_file_engine.TotalTraces, position=1)
		should_i_run = True
		while should_i_run:

			# check if the smaller file engine is done
			if trace_counter < smaller_file_engine.TotalTraces:
				# Set the next trace
				trace_meta = smaller_file_engine[trace_counter]

				# Get the trace from the smaller file engine
				trace = trace_meta[0]
				
				# Get metadata vector 
				tupla = trace_meta[1]

				# Get the label of the trace
				label = smaller_file_engine.Label
			
				# Reshape to apply scalers
				trace = trace.reshape(1, -1)
				# Apply scalers
					# Apply standard scaler
				trace = self.__des_scalers_list[0].transform(trace)
					# Apply minmax scaler
				trace = self.__des_scalers_list[1].transform(trace)
				# Reshape again for vortex
				trace = trace.reshape(-1, 1)
				# Sinking to the vortex
				vortex = ArrayVortex(FormatConfig.FROM_ARRAY_TO_1DDLMODEL)
				trace  = vortex.Sinking(trace)

				# Prepare dict for input data for both autoencoder and encoder
				predict_dict = {'encoder_input': trace}

				# Make predictions (autoencoder and encoder)
				predictions_enc = self.__encoder_model.predict(predict_dict)

				# Reverse vortex
				vortex    = ArrayVortex(FormatConfig.FROM_1DDLMODEL_TO_ARRAY)
				trace_enc = vortex.Sinking(predictions_enc)

				# Feed ascadBuilder
				self.__ascadBuilder.Feed_Traces(atraces=[trace_enc], ametadata=tupla, labeler=[label])
				pbar1.update()
			
			if trace_counter < bigger_file_engine.TotalTraces:
				# Set the next trace
				trace_meta = bigger_file_engine[trace_counter]

				# Get the trace from the smaller file engine
				trace = trace_meta[0]
				
				# Get metadata vector 
				tupla = trace_meta[1]

				# Get the label of the trace
				label = smaller_file_engine.Label
			
				# Reshape to apply scalers
				trace = trace.reshape(1, -1)
				# Apply scalers
					# Apply standard scaler
				trace = self.__des_scalers_list[0].transform(trace)
					# Apply minmax scaler
				trace = self.__des_scalers_list[1].transform(trace)
				# Reshape again for vortex
				trace = trace.reshape(-1, 1)
				# Sinking to the vortex
				vortex = ArrayVortex(FormatConfig.FROM_ARRAY_TO_1DDLMODEL)
				trace  = vortex.Sinking(trace)

				# Prepare dict for input data for both autoencoder and encoder
				predict_dict = {'encoder_input': trace}

				# Make predictions (autoencoder and encoder)
				predictions_enc = self.__encoder_model.predict(predict_dict)

				# Reverse vortex
				vortex    = ArrayVortex(FormatConfig.FROM_1DDLMODEL_TO_ARRAY)
				trace_enc = vortex.Sinking(predictions_enc)

				# Feed ascadBuilder
				self.__ascadBuilder.Feed_Traces(ptraces=[trace_enc], pmetadata=tupla, labeler=[label])
				pbar2.update()
			else:
				should_i_run = False

			trace_counter = trace_counter + 1

	def __two_file_engine_function_batches(self):

		batches_index = 0
		smaller_batches_index = 0
		
		if self.__fengine_total_traces is None:
			self.__fengine_total_traces = self.__file_engine.TotalTraces
		if self.__second_fengine_total_traces is None:
			self.__second_fengine_total_traces = self.__second_file_engine.TotalTraces
		
		smaller_file_engine = self.__file_engine if self.__fengine_total_traces < self.__second_fengine_total_traces else self.__second_file_engine
		bigger_file_engine  = self.__file_engine if self.__fengine_total_traces > self.__second_fengine_total_traces else self.__second_file_engine

		# Check if the number of batches is fixed contained in total traces
		bigger_file_engine_traces = self.__fengine_total_traces if self.__fengine_total_traces > self.__second_fengine_total_traces else self.__second_fengine_total_traces
		# Fixed number of batches for bigger file engine
		bigger_fe_number_batches = int(bigger_file_engine_traces / self.__batch_size) if (bigger_file_engine_traces % self.__batch_size) == 0 else int((bigger_file_engine_traces - (bigger_file_engine_traces % self.__batch_size)) / self.__batch_size)

		# Check if the number of batches is fixed contained in total traces of the smaller file engine
		smaller_file_engine_traces = self.__fengine_total_traces if self.__fengine_total_traces < self.__second_fengine_total_traces else self.__second_fengine_total_traces
		# Fixed number of batches for smaller file engine
		smaller_fe_number_batches = int(smaller_file_engine_traces / self.__batch_size) if (smaller_file_engine_traces % self.__batch_size) == 0 else int((smaller_file_engine_traces - (smaller_file_engine_traces % self.__batch_size)) / self.__batch_size)

		pbar1 = tqdm(total=bigger_file_engine_traces, position=0, leave=True)
		pbar2 = tqdm(total=smaller_file_engine_traces, position=1, leave='output')
		should_i_run = True
		while should_i_run:

			# check if the smaller file engine is done
			if batches_index < smaller_fe_number_batches:
				# Set the next trace
				trace_meta = np.array(smaller_file_engine[batches_index * self.__batch_size:(batches_index + 1) * self.__batch_size], dtype=np.ndarray)

				# Get the trace from the smaller file engine
				traces = np.vstack(trace_meta[:,0])

				# Get metadata vector 
				tuplas = np.vstack(trace_meta[:,1])

				# Get the label of the trace
				labels = trace_meta[:,2]
			
				traces = self.__des_scalers_list[0].transform(traces)
				traces = self.__des_scalers_list[1].transform(traces)

				predict_dict = {'encoder_input': traces}
				predictions_enc = self.__encoder_model.predict(predict_dict)

				if (len(predictions_enc.shape) == 3):
					predictions_enc = np.squeeze(predictions_enc)

				self.__ascadBuilder.Feed_Batch_Traces(self.__batch_size, atraces=predictions_enc, ametadata=tuplas, labeler=labels, flush_flag=1000)
				# Update batches index of the smaller fileengine
				smaller_batches_index = smaller_batches_index + 1

				pbar1.update(self.__batch_size)
			
			if batches_index < bigger_fe_number_batches:
				# Set the next trace
				trace_meta = np.array(bigger_file_engine[batches_index * self.__batch_size:(batches_index + 1) * self.__batch_size], dtype=np.ndarray)

				# Get the trace from the smaller file engine
				traces = np.vstack(trace_meta[:,0])
				
				# Get metadata vector 
				tuplas = np.vstack(trace_meta[:,1])

				# Get the label of the trace
				labels = trace_meta[:,2]
			
				traces = self.__des_scalers_list[0].transform(traces)
				traces = self.__des_scalers_list[1].transform(traces)
				
				predict_dict = {'encoder_input': traces}
				predictions_enc = self.__encoder_model.predict(predict_dict)

				self.__ascadBuilder.Feed_Batch_Traces(self.__batch_size, ptraces=predictions_enc, pmetadata=tuplas, labeler=labels, flush_flag=1000)
				pbar2.update(self.__batch_size)
			else:
				should_i_run = False
			# Update batches index of the bigger fileengine
			batches_index = batches_index + 1
		
		# Check if the number of batches is fixed contained in total traces
		if (bigger_file_engine_traces % self.__batch_size) != 0:
			# Set the next trace
			trace_meta = np.array(bigger_file_engine[(batches_index - 1) * self.__batch_size:(((batches_index - 1) * self.__batch_size) + (bigger_file_engine_traces % self.__batch_size))], dtype=np.ndarray)
			# Get the trace from the smaller file engine
			traces = np.vstack(trace_meta[:,0])
			# Get metadata vector 
			tuplas = np.vstack(trace_meta[:,1])
			# Get the label of the trace
			labels = trace_meta[:,2]
			traces = self.__des_scalers_list[0].transform(traces)
			traces = self.__des_scalers_list[1].transform(traces)
			predict_dict = {'encoder_input': traces}
			predictions_enc = self.__encoder_model.predict(predict_dict)
			self.__ascadBuilder.Feed_Batch_Traces(self.__batch_size, ptraces=predictions_enc, pmetadata=tuplas, labeler=labels, flush_flag=None)
			pbar2.update((bigger_file_engine_traces % self.__batch_size))

		# Check if the number of batches is fixed contained in total traces of the smaller file engine
		if (smaller_file_engine_traces % self.__batch_size) != 0:
			# Set the next trace
			trace_meta = np.array(smaller_file_engine[(smaller_batches_index - 1) * self.__batch_size:(((smaller_batches_index - 1) * self.__batch_size) + (smaller_file_engine_traces % self.__batch_size))], dtype=np.ndarray)
			# Get the trace from the smaller file engine
			traces = np.vstack(trace_meta[:,0])
			# Get metadata vector 
			tuplas = np.vstack(trace_meta[:,1])
			# Get the label of the trace
			labels = trace_meta[:,2]
			traces = self.__des_scalers_list[0].transform(traces)
			traces = self.__des_scalers_list[1].transform(traces)
			predict_dict = {'encoder_input': traces}
			predictions_enc = self.__encoder_model.predict(predict_dict)
			self.__ascadBuilder.Feed_Batch_Traces(self.__batch_size, atraces=predictions_enc, ametadata=tuplas, labeler=labels, flush_flag=None)
			pbar1.update((smaller_file_engine_traces % self.__batch_size))
		
		print ()

	def __del__(self):
		# closing        
		if self.__ascadBuilder is not None:
			self.__ascadBuilder.Close_File()
	
	def close_created_file(self):
		if self.__ascadBuilder is not None:
			self.__ascadBuilder.close()