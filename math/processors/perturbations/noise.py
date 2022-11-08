import numpy as np
from sfileengine import FileEngine
from sfileengine import H5FileEngine
from sutils import trange
from builders import ASCADBuilder
from builders import TRSBuilder
import os

def _h5file_engine_gaussian_noiser(file_engine:FileEngine, attack_file_engine:FileEngine=None, noise_level:int=1, output_path:str=None):
	print('[INFO -AddGaussianNoise-]:', noise_level, 'amplitude')
	output_path = output_path if output_path is not None else file_engine.Path
	output_path = os.path.join(output_path, '{}_noisy_level_{}.h5'.format(file_engine.FileName, noise_level))
	m_ascad_builder = ASCADBuilder(output_path, file_engine.MetadataTypeDescriptor)
	m_ascad_builder.Set_Profiling(file_engine.TotalSamples)
	m_ascad_builder.Set_Attack(file_engine.TotalSamples)
	m_ascad_builder.Add_Attack_Label()
	m_ascad_builder.Add_Profiling_Label()

	larger_limit = file_engine.TotalTraces if file_engine.TotalTraces > attack_file_engine.TotalTraces else attack_file_engine.TotalTraces
	for trace in trange(larger_limit, desc='[INFO -AddGaussianNoise-]: Creating noisy traces'):

		if trace <= file_engine.TotalTraces:
			# Set the next trace
			trace_meta = file_engine[trace]
			# Get the trace from the smaller file engine
			traces = trace_meta[0]
			# Get metadata vector 
			tuplas = trace_meta[1]
			# Get the label of the trace
			labels = trace_meta[2]

			tuplas = np.array([tuplas], dtype=file_engine.MetadataTypeDescriptor)
			
			noise = np.random.normal(0, noise_level, size=np.shape(traces))
			m_ascad_builder.Feed_Traces(ptraces=[noise], pmetadata=tuplas, labeler=[labels], flush_flag=1000)

		if trace <= attack_file_engine.TotalTraces:
			# Set the next trace
			trace_meta = attack_file_engine[trace]
			# Get the trace from the smaller file engine
			traces = trace_meta[0]
			# Get metadata vector 
			tuplas = trace_meta[1]
			# Get the label of the trace
			labels = trace_meta[2]

			tuplas = np.array([tuplas], dtype=file_engine.MetadataTypeDescriptor)
			
			noise = np.random.normal(0, noise_level, size=np.shape(traces))
			m_ascad_builder.Feed_Traces(atraces=[noise], ametadata=tuplas, labeler=[labels], flush_flag=1000)
	
	m_ascad_builder.Close_File()

def add_gussian_noise(file_engine:FileEngine, attack_file_engine:FileEngine=None, noise_level:int=1, output_path:str=None):
	
	if noise_level < 1:
		print('[INFO -AddGaussianNoise-]: Noise level has to be bigger than 1')
	else:
		if attack_file_engine is not None:
			_h5file_engine_gaussian_noiser(file_engine, attack_file_engine, noise_level, output_path)
