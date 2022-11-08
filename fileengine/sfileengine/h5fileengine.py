"""
TODO: Escribir los nombres de los archivos con __<nombre>__.py
para ver si estos se ocultan cuando se referencia a las clases y metodos
del archivo

"""
import numpy as np
from .fileengine import FileEngine
from builders    import load_ascad_handler
from builders 	 import TRSBuilder
from sutils 	 import trange
import os

class H5FileEngine(FileEngine):
	def __init__(self, h5_file, group='/', mode='r', metadata_type=np.uint8, label_iteration=False, 
	             explicit_nodes=(None, None, None), processing_function=None, trace_processor=None):
		"""
		docstring
		"""	
		if isinstance(h5_file, str):
			pointer, path      = load_ascad_handler(h5_file, mode=mode)
			if pointer is None:
				raise FileNotFoundError('[ERROR -H5FileEngine-]: The specified h5 file does not exist')
			self._file_pointer = pointer
			self._file         = path
			
		else:	
			self._file_pointer = h5_file
			self._file         = h5_file.filename
			
		self.__label    = None
		self.__metadata = None
		self.__traces   = None
		if group[0] != '/':
			self.__group = group
			group = '/'+group
		else:
			self.__group = group[1:]

		for node in self.H5File.list_nodes(group):
			if 'label' in str(node) and explicit_nodes[2] is None: 
				self.__label = node
			elif 'metadata' in str(node) and explicit_nodes[1] is None:
				self.__metadata = node
			elif 'traces' in str(node) and explicit_nodes[0] is None:
				self.__traces = node
	
		if explicit_nodes[2] is not None:
			self.__label = self.H5File.get_node(explicit_nodes[2])
		if explicit_nodes[1] is not None: 
			self.__metadata = self.H5File.get_node(explicit_nodes[1])
		if explicit_nodes[0] is not None:
			self.__traces = self.H5File.get_node(explicit_nodes[0])
		
		if not label_iteration:
			self.__iteration_property = self.__class__.Data.fget
		else:
			self.__iteration_property = self.__class__.Label.fget
		
		if processing_function is not None and isinstance(processing_function, str):
			if processing_function == 'merging':
				from .processors import merging
				processing_function = merging
		
		total_traces_function = self.__total_traces
		total_samples_function = self.__total_samples
		if trace_processor is not None:
			from .processors import ProcessingEngine

			if isinstance(trace_processor, str):
				if trace_processor == 'abs':
					from .processors import abs
					trace_processor = abs
			elif isinstance(trace_processor, ProcessingEngine):
				
				if trace_processor.TotalTraces is not None:
					total_traces_function = trace_processor.TotalTraces
					
				if trace_processor.TotalSamples is not None:
					total_samples_function = trace_processor.TotalSamples
				
				trace_processor = trace_processor.run
			else:
				print ('[WARNING -H5FileEngine-]: trace_processor is not None, but it is not a identified trace processor')

		self.__total_traces_function = total_traces_function
		self.__total_samples_function = total_samples_function
		self.__processing_function = processing_function if processing_function is not None else self.__identity_preprocesing
		self.__traces_processor    = trace_processor if trace_processor is not None else self.__identity_traces_processing
		self.TracePointer    = 0
		self.__metadata_type = metadata_type
		self._file_type 	 = 'h5'

	def __identity_preprocesing(self, data):
		return data

	def __identity_traces_processing(self, trace):
		return trace

	@property
	def __total_traces(self):
		_ = self.__traces[0]
		return self.__traces.nrows

	@property
	def __total_samples(self):
		_ = self.__traces[0]
		return self.__traces.shape[1]

	@property
	def MetadataTypeDescriptor(self):
		if self.__metadata is not None:
			return self.__metadata[0].dtype
		return None

	@property
	def TotalTraces(self) -> int:
		return self.__total_traces_function

	@property
	def TotalSamples(self) -> int:
		return self.__total_samples_function

	@property
	def Group(self):
		return self.__group

	@property
	def OriginalPath(self):
		return self.H5File.filename

	@property
	def H5File(self):
		"""
		docstring
		"""
		return self.FilePointer
	
	@property
	def Label(self):
		return self.__label[self.TracePointer]

	@property
	def Trace(self):
		"""
		docstring
		"""
		if self.__label is None:
			return [self.__traces_processor(self.__traces[self.TracePointer]), self.__iteration_property(self)]
		else:
			return [self.__traces_processor(self.__traces[self.TracePointer]), self.__iteration_property(self), self.__label[self.TracePointer]]
	
	@property
	def Data(self):	
		#return np.concatenate(list(self.__metadata[self.TracePointer]), axis=0)
		return self.__processing_function(self.__metadata[self.TracePointer])

	def to_trsfile(self, suffix:str=None):
		from .processors import merging
		chose_processor = self.__processing_function
		if not (chose_processor == merging):
			chose_processor = merging

		trsBuilder = TRSBuilder(os.path.join(self.Path, '{}{}.trs'.format(self.FileName, suffix if not None else '')))
		for _ in trange(self.TotalTraces, desc='[INFO -H5FileEngine-]: Converting to trsfile'):
			trace = self.__traces_processor(self.__traces[self.TracePointer])
			meta = chose_processor(self.__metadata[self.TracePointer])
			#trace = np.array(trace, dtype=np.ndarray)
			
			meta_label = np.zeros(shape=(meta.shape[0] + 1), dtype=meta.dtype)
			meta_label[0] = self.__label[self.TracePointer]
			meta_label[1:] = meta
			trsBuilder.Feed_Traces(trace, meta_label)

			self.NextTrace


	def __del__(self):
		self.__label    = None
		self.__metadata = None
		self.__traces   = None
		self._file_pointer = None
		self._file         = None
	
	"""
	TODO: Hay un error cuando un elemento de la metadata no es un vector, i.e. si un escalar es parte de la metadata
	entonces np.concatenate no funciona, esto es porque el escalar no tiene shape.
	La forma en la que lo solucioné fue recortando el ultimo elemento en ASCAD que representa la desyncronizacion
	pero esto no es muy eficaz:
	@property
    def Data(self):
		return np.concatenate(list(self.__metadata[self.TracePointer])[:-1], axis=0)

	ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 4 has 0 dimension(s)
	"""
