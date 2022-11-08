from builders import TRSBuilder
import numpy as np
import os
from tqdm import trange
from trsfile import Header
import sutils

##################################################################################
def scalar_product(trs_file, scalar):
	
	# Get number of samples and metadata legth
	n_traces = len(trs_file)

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-scalarproduct-{}.trs'.format(fname, scalar))

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('scalar', scalar)
	jsonParameter.add_string('original_file', original_path)

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'scalarproduct')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'scalarproduct')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs TRSBuilder
	trsBuilder = TRSBuilder(final_path, description=new_description)

	# Executing
	for i in trange(n_traces, desc='[INFO *ScalarProduct*]: Computing scalar product'):        
		trsBuilder.Feed_Traces(np.array(trs_file[i]) * scalar, np.frombuffer(trs_file[i].data, dtype='uint8'))
	
	print('[INFO *ScalarProduct*]: Closing file')
	trsBuilder.Close_File()
	print('[INFO *ScalarProduct*]: Done')
	return final_path
##################################################################################
def scalar_addition(trs_file, scalar):
	# Get number of samples and metadata legth
	n_traces = len(trs_file)

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-scalaraddition-{}.trs'.format(fname, scalar))

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('scalar', scalar)
	jsonParameter.add_string('original_file', original_path)

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'scalaraddition')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'scalaraddition')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs TRSBuilder
	trsBuilder = TRSBuilder(final_path, description=new_description)

	# Executing
	for i in trange(n_traces, desc='[INFO *ScalarAddition*]: Computing scalar addition'):        
		trsBuilder.Feed_Traces(np.array(trs_file[i]) + scalar, np.frombuffer(trs_file[i].data, dtype='uint8'))
	
	print('[INFO *ScalarAddition*]: Closing file')
	trsBuilder.Close_File()
	print('[INFO *ScalarAddition*]: Done')
	return final_path
##################################################################################
def vector_addition_numpy(trs_file, numpy_vector, sustract=False):
	# Sanity checks
	if (len(numpy_vector) > 1):
		print ('[WARNING *VectorAdditionNumpy*]: NumpyVector should contain only one trace, firts element is going to be taken')
	if len(trs_file[0]) != len(numpy_vector):
		print ('[ERROR *VectorAdditionNumpy*]: Vector\'s different dimensions, behaviour not supported, returning None')
		return None

	# Get number of samples and metadata legth
	n_traces = len(trs_file)

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-vectoradditionnumpy-{}.trs'.format(fname, scalar))

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('trsfile2', trsfile2path)

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'vectoradditionnumpy')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'vectoradditionnumpy')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs TRSBuilder
	trsBuilder = TRSBuilder(final_path, description=new_description)

	# Executing
	for i in trange(n_traces, desc='[INFO *VectorAdditionNumpy*]: Computing vector addition numpy{}'.format('(sustract active)' if sustract else '')):        
		trsBuilder.Feed_Traces(np.array(trs_file[i]) + ((-1) * numpy_vector) if sustract else numpy_vector, np.frombuffer(trs_file[i].data, dtype='uint8'))
	
	print('[INFO *VectorAdditionNumpy*]: Closing file')
	trsBuilder.Close_File()
	print('[INFO *VectorAdditionNumpy*]: Done')
	return final_path
##################################################################################
def vector_addition(trs_file, trs_file2, sustract=False):
	# Sanity check
	if (len(trs_file2) > 1):
		print ('[WARNING *VectorAddition*]: TrsFile2 should contain only one trace, firts trace is going to be taken')
	if len(trs_file[0]) != len(trs_file2[0]):
		print ('[ERROR *VectorAdditionNumpy*]: Vector\'s different dimensions, behaviour not supported, returning None')
		return None

	# Get number of samples and metadata legth
	n_traces = len(trs_file)

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-vectoraddition-{}.trs'.format(fname, scalar))

	# Geting trsf2path
	trsfile2path = trs_file2.engine.path

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('trsfile2', trsfile2path)
	jsonParameter.add_string('original_file', original_path)

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'vectoraddition')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'vectoraddition')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs TRSBuilder
	trsBuilder = TRSBuilder(final_path, description=new_description)

	# Executing
	for i in trange(n_traces, desc='[INFO *VectorAddition*]: Computing vector addition{}'.format('(sustract active)' if sustract else '')):        
		trsBuilder.Feed_Traces(np.array(trs_file[i]) + ((-1) * np.array(trs_file2[0])) if sustract else np.array(trs_file2[0]), np.frombuffer(trs_file[i].data, dtype='uint8'))
	
	print('[INFO *VectorAddition*]: Closing file')
	trsBuilder.Close_File()
	print('[INFO *VectorAddition*]: Done')
	return final_path

##################################################################################
def vector_abs(trs_file):
	# Get number of samples and metadata length
	n_traces = len(trs_file)

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-abs.trs'.format(fname))

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('original_file', original_path)

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'abs')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'abs')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs TRSBuilder
	trsBuilder = TRSBuilder(final_path, description=new_description)

	# Executing
	for i in trange(n_traces, desc='[INFO *Abs*]: Computing Abs trace'):
		trsBuilder.Feed_Traces(np.abs(trs_file[i]), np.frombuffer(trs_file[i].data, dtype='uint8'), title=trs_file[i].title)
	
	print('[INFO *Abs*]: Closing file')
	trsBuilder.Close_File()
	print('[INFO *Abs*]: Done')
	return final_path