from .type_descriptor import MetadataTypeDescriptor
from .ascad_group import ASCADGroupType
from .ascad_group import Profiling
from .ascad_group import Attack
from .loaders import load_ascad_handler
import os
import tables
import numpy as np
##################################################################################
"""
def __narrow_traces_from_trs(trfile_stream, 
						   ascad_group: ASCADGroup, 
						   classes_index: dict, 
						   feature_limits: tuple=None):
	
	'''
	Narrows traces from the specified trs_file_stream and allocates them into
	the specified ascad group
	
	TODO: Complete this docs
	Args:
		trfile_stream (trs stream): The stream of the trsfile used to collect the traces
		ascad_group (ASCADGroup) : The ascad group where the traces will be saved.        
	'''
	
	print ('[INFO]: Narrow traces from trs')
	for i, (label, index_list) in enumerate(classes_index.items()):
		print ('[INFO]: Getting taces from class: {}'.format(label))
		number_traces = len(index_list)
		for j in trange(number_traces, desc='[INFO]: Narrowing traces'):
			ascad_group.Labels.append([label])
			trace = trfile_stream[index_list[j]].samples
			if feature_limits is not None:
				trace = trace[feature_limits[0]:feature_limits[1]]
			
			trace = trace.reshape(1, -1)
				
			ascad_group.Traces.append(trace)
			ascad_group.h5Stream.flush()
"""
##################################################################################
class ASCADBuilder():
	def __init__(self, output_path, descriptor:np.dtype, compress=True):
		"""Builds a ASCAD type of file for storagind traces.

		Args:
			output_path (str): path and name of the new ASCAD file.
			descriptor (np.dtype): metadata type descriptor to build the metadata table of the ASCAD file.
			compress (bool): whether the file will be compressed. Bear in mind that Tables module is 
				in charge of the compression procedure, it has some dependencies, when those are not
				installed the compression will be omited.
		"""

		# Check if stream is already open
		(old_h5_stream, _) = load_ascad_handler(output_path, mode='w', force_mode=True)
		if old_h5_stream is not None:
			self.__h5_stream = old_h5_stream
		else:
			# Crear el stream from tables
			if compress:
				FILTERS = tables.Filters(complib='zlib', complevel=5)
				self.__h5_stream = tables.open_file(output_path, mode='w', filters=FILTERS)
			else:
				self.__h5_stream = tables.open_file(output_path, mode='w')
		
		# Create attack group and its helpers
		self.__attack_group = None
		self.__attack_trace_limits = None
		self.__attack_trace_trsfile = None

		# Create profiling group and its helpers
		self.__profiling_group = None
		self.__profiling_trace_limits = None
		self.__profiling_trace_trsfile = None

		# set descriptor
		self.__descriptor = descriptor

		# set utilities
		self.__number_atraces = 0
		self.__number_ptraces = 0
		self.__closed = False
	
	@property
	def ATraces_Fed(self):
		return self.__number_atraces
	
	@property
	def PTraces_Fed(self):
		return self.__number_ptraces

	@property
	def Descriptor(self):
		"""
		docstring
		"""
		return self.__descriptor

	def Set_Profiling(self, data_legth, dtype=np.float):
		"""
		docstring
		"""
		(pgroup, pTraces, pMetadata) = self.__create_profiling_set(self.__h5_stream, data_legth, dtype)
		self.__profiling_group = Profiling(self.__h5_stream, pgroup, traces=pTraces, metadata=pMetadata)

	def Add_Profiling_Label(self):
		if self.__profiling_group is not None:
			profiling_traces_label = self.__h5_stream.create_earray(self.__profiling_group.Group, 'labels', obj=np.zeros(shape=(0, ), dtype=np.uint8))
			self.__profiling_group.Label = profiling_traces_label

	def Set_Attack(self, data_legth, dtype=np.float):
		"""
		docstring
		"""
		(agroup, aTraces, aMetadata) = self.__create_attack_set(self.__h5_stream, data_legth, dtype)
		self.__attack_group = Attack(self.__h5_stream, agroup, traces=aTraces, metadata=aMetadata)
	
	def Add_Attack_Label(self):
		if self.__attack_group is not None:
			attack_traces_label = self.__h5_stream.create_earray(self.__attack_group.Group, 'labels', obj=np.zeros(shape=(0, ), dtype=np.uint8))
			self.__attack_group.Label = attack_traces_label

	def Feed_Traces(self, ptraces=None, atraces=None, pmetadata=None, ametadata=None, labeler=None, flush_flag=None):
		"""Feed traces into the H5 File (ASCAD structured)
		Metadata should be in the shape: np.array([(trace.textin, trace.key, trace.textout, mask, fr_index)], dtype=descriptor)
		Trace should be in the shape: [np.array(trace.wave, dtype='float32')]
		Label should be in the shape: [label]
		"""
		if ptraces is not None and self.__profiling_group is not None:
			self.__profiling_group.Traces.append(ptraces)
			self.__profiling_group.MetaDataTable.append(pmetadata)
			if labeler is not None and self.__profiling_group.Label is not None:
				self.__profiling_group.Label.append(labeler)
			self.__number_ptraces += 1
			
		if atraces is not None and self.__attack_group is not None:
			self.__attack_group.Traces.append(atraces)
			self.__attack_group.MetaDataTable.append(ametadata)
			if labeler is not None and self.__attack_group.Label is not None:
				self.__attack_group.Label.append(labeler)
			self.__number_atraces += 1

		if flush_flag is not None:
			if (self.__number_ptraces % flush_flag) == 0 or (self.__number_atraces % flush_flag) == 0:
				self.__h5_stream.flush()
	
	def Feed_Batch_Traces(self, batch_size, ptraces=None, atraces=None, pmetadata=None, ametadata=None, labeler=None, flush_flag=None):
		"""Feed traces into the H5 File (ASCAD structured)
		Metadata should be in the shape: np.array([(trace.textin, trace.key, trace.textout, mask, fr_index)], dtype=descriptor)
		Trace should be in the shape: [np.array(trace.wave, dtype='float32')]
		"""
		if ptraces is not None and self.__profiling_group is not None:
			self.__profiling_group.Traces.append(ptraces)
			self.__profiling_group.MetaDataTable.append(pmetadata)
			if labeler is not None and self.__profiling_group.Label is not None:
				self.__profiling_group.Label.append(labeler)
			self.__number_ptraces += batch_size
			
		if atraces is not None and self.__attack_group is not None:
			self.__attack_group.Traces.append(atraces)
			self.__attack_group.MetaDataTable.append(ametadata)
			if labeler is not None and self.__attack_group.Label is not None:
				self.__attack_group.Label.append(labeler)
			self.__number_atraces += batch_size

		if flush_flag is not None:
			if (self.__number_ptraces % (flush_flag + batch_size)) == 0 or (self.__number_atraces % (flush_flag + batch_size)) == 0:
				self.__h5_stream.flush()

	def __create_profiling_set(self, h5_stream, data_legth, descriptor, dtype=None):
		"""
		Args:
			h5_stream (tables stream): Stream from tables file.
			
			data_legth (int): Lenght of the trace.

			descriptor (MetadataTypeDescriptor): Metadata descriptor.
			
			dtype (np type): Specify the type of the traces data.
			
		Returns:
			A tuple (2D-tuple): profiling_traces_label, profiling_traces_traces.
		"""
		#Check whether the file already has a group called Profiling_traces 
		if '/Profiling_traces' in str(self.__h5_stream.list_nodes('/')):
			self.__h5_stream.remove_node('/Profiling_traces', recursive=True)
			
		profiling_traces = self.__h5_stream.create_group(self.__h5_stream.root, "Profiling_traces")
		#profiling_traces_label = h5_stream.create_earray(profiling_traces, 'label', obj=np.zeros(shape=(0, ), dtype=np.uint8))
		profiling_traces_traces = self.__h5_stream.create_earray(profiling_traces, 'traces', obj=np.zeros(shape=(0, data_legth), dtype=dtype))
		profiling_metadata = self.__h5_stream.create_table(profiling_traces, 'metadata', createparents=False, description=self.__descriptor)
		
		return (profiling_traces, profiling_traces_traces, profiling_metadata)

	def __create_attack_set(self, h5_stream, data_legth, dtype=None):
		"""
		Args:
			h5_stream (tables stream): Stream from tables file.
			
			data_legth (int): Lenght of the trace.
			
			descriptor (MetadataTypeDescriptor): Metadata descriptor.
			
			dtype (np type): Specify the type of the traces data.
			
		Returns:
			A tuple (2D-tuple): profiling_traces_label, profiling_traces_traces.
		"""
		
		#Check whether the file already has a group called Attack_traces 
		if '/Attack_traces' in str(self.__h5_stream.list_nodes('/')):
			self.__h5_stream.remove_node('/Attack_traces', recursive=True)

		attack_traces = self.__h5_stream.create_group(self.__h5_stream.root, "Attack_traces")
		attack_traces_traces = self.__h5_stream.create_earray(attack_traces, 'traces', obj=np.zeros(shape=(0, data_legth), dtype=dtype))
		attack_metadata = self.__h5_stream.create_table(attack_traces, 'metadata', createparents=False, description=self.__descriptor)

		return (attack_traces, attack_traces_traces, attack_metadata)

	def __del__(self):
		if not self.__closed:
			self.Close_File()

	def Close_File(self):
		self.__closed = True
		self.__h5_stream.close()

	def close(self):
		self.__closed = True
		self.__h5_stream.close()

##################################################################################
"""
class ASCADCreator():
	def __init__(self):
		self.__h5_stream = None
		self.__profiling = None
		self.__attack = None
	
	@property
	def Stream(self):
		return self.__h5_stream

	@property
	def ProfilingTraces(self):
		return self.__profiling.Traces

	@property
	def ProfilingLabel(self):
		return self.__profiling.Labels
	
	@property
	def AttackTraces(self):
		return self.__attack.Traces

	@property
	def AttackLabel(self):
		return self.__attack.Labels

	def create(self, output_path, data_legth, dtype):
		(stream, pLabels, pTraces, aLabels, aTraces) = __create_ascad_dataset(output_path, data_legth, dtype)
		self.__h5_stream = stream
		self.__profiling = Profiling(stream, pLabels, pTraces)
		self.__attack = Attack(stream, aLabels, aTraces)
			
	def fullfil_profiling(self, trfile_stream, classes_index, feature_limits=None, apply_standard_scaler=True, classification=None):
		from .. import narrow_traces_from_trs
		narrow_traces_from_trs(trfile_stream, self.__profiling, classes_index, feature_limits, apply_standard_scaler, classification=classification)

	def fullfil_attack(self, trfile_stream, classes_index, feature_limits=None, apply_standard_scaler=True, classification=None):
		from .. import narrow_traces_from_trs
		narrow_traces_from_trs(trfile_stream, self.__attack, classes_index, feature_limits, apply_standard_scaler, classification=classification)
	
	def fullfil_profiling_with_prescaled(self, trfile_stream, classification, classes_index, feature_limits=None):
		from .. import narrow_trs_using_prescaled
		narrow_trs_using_prescaled(trfile_stream, classification, self.__profiling, classes_index, feature_limits)

	def fullfil_attack_with_prescaled(self, trfile_stream, classification, classes_index, feature_limits=None):
		from .. import narrow_trs_using_prescaled
		narrow_trs_using_prescaled(trfile_stream, classification, self.__attack, classes_index, feature_limits)

	def __fullfil_profiling(self, trfile_stream, classes_index, feature_limits=None):
		__narrow_traces_from_trs(trfile_stream, self.__profiling, classes_index, feature_limits)

	def __fullfil_attack(self, trfile_stream, classes_index, feature_limits=None):
		__narrow_traces_from_trs(trfile_stream, self.__attack, classes_index, feature_limits)

	def fullfil_profiling_by_rsource(self, trs_resource, classes_index, feature_limits=None, apply_standard_scaler=True, classification=None):
		from .. import narrow_traces_from_trs
		import trsfile
		if trs_resource.FileType is not 'trsfile':
			print ('[WARNING]: Apparently, the resources is not a trsfile')
		
		trfile_stream = trsfile.open(trs_resource.Location, mode='r')
		self.narrow_traces_from_trs(trfile_stream, self.__profiling, classes_index, feature_limits, apply_standard_scaler, classification=classification)

	def fullfil_attack_by_rsource(self, trs_resource, classes_index, feature_limits=None, apply_standard_scaler=True, classification=None):
		from .. import narrow_traces_from_trs
		import trsfile
		if trs_resource.FileType is not 'trsfile':
			print ('[WARNING]: Apparently, the resources is not a trsfile')
		
		trfile_stream = trsfile.open(trs_resource.Location, mode='r')
		self.narrow_traces_from_trs(trfile_stream, self.__attack, classes_index, feature_limits, apply_standard_scaler, classification=classification)
	
	def fullfil_profiling_by_rsource_with_prescaled(self, trs_resource, classification, classes_index, feature_limits=None):
		from .. import narrow_trs_using_prescaled
		import trsfile
		if trs_resource.FileType is not 'trsfile':
			print ('[WARNING]: Apparently, the resources is not a trsfile')
		
		trfile_stream = trsfile.open(trs_resource.Location, mode='r')
		self.narrow_trs_using_prescaled(trfile_stream, classification, self.__profiling, classes_index, feature_limits)

	def fullfil_attack_by_rsource_with_prescaled(self, trs_resource, classification, classes_index, feature_limits=None):
		from .. import narrow_trs_using_prescaled
		import trsfile
		if trs_resource.FileType is not 'trsfile':
			print ('[WARNING]: Apparently, the resources is not a trsfile')
		
		trfile_stream = trsfile.open(trs_resource.Location, mode='r')
		self.narrow_trs_using_prescaled(trfile_stream, classification, self.__attack, classes_index, feature_limits)


	def close(self):
		self.__h5_stream.close()

"""