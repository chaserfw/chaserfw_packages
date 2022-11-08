from sstats import compute_samples_mean_trs
from sstats import compute_samples_std_trs
from trsbuilder import Creator
from trsfile import Header
from tqdm import trange
import numpy as np
import sutils
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

##################################################################################
def Standard_Scaler(trs_file, trs_mean_file=None, trs_std_file=None):
	"""Compute Standard Scale to the trs file
	z = (x - u) / s
	where: 
		z: it's the new standard scaled trace.
		u: it's the mean of the trace set
		s: it's the standard deviation of the trace set
	"""

	# Get number of traces
	n_traces = len(trs_file)
	samples_mean = compute_samples_mean_trs(trs_file, n_traces) if trs_mean_file is None else trs_mean_file[0]
	samples_std = compute_samples_std_trs(trs_file, n_traces, samples_mean) if trs_std_file is None else trs_std_file[0]

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-standardscaler.trs'.format(fname))

	# Getting path of trs_mean_file and trs_std_file (if they aren't None)
	trs_mean_path = trs_mean_file.engine.path if trs_mean_file is not None else None
	trs_std_path  = trs_std_file.engine.path  if trs_std_file is not None else None 

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('trs_mean_file', trs_mean_path if trs_mean_path is not None else '')
	jsonParameter.add_string('original_file', original_path if trs_std_path is not None else '')

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'standardscaler')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'standardscaler')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs creator
	creator = Creator(final_path, description=new_description)

	# Executing
	for i in trange(n_traces, desc='[INFO *StandardScaler*]: Computing standard scaler'):        
		creator.Feed_Traces(np.true_divide((np.array(trs_file[i]) - samples_mean), samples_std), np.frombuffer(trs_file[i].data, dtype='uint8'))
	
	print('[INFO *StandardScaler*]: Closing file')
	creator.Close_File()
	if trs_mean_file is not None:
		print('[INFO *StandardScaler*]: Closing trs mean file')
		trs_mean_file.close()
	if trs_std_file is not None:
		print('[INFO *StandardScaler*]: Closing trs std file')
		trs_std_file.close()

	print('[INFO *StandardScaler*]: Done')
	return final_path
##################################################################################
def Standard_Scaler_SK(trs_file):
	"""Compute Standard Scale to the trs file
	z = (x - u) / s
	where: 
		z: it's the new standard scaled trace.
		u: it's the mean of the trace set
		s: it's the standard deviation of the trace set
	"""

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-skstandardscaler.trs'.format(fname))

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('trs_mean_file', "None")
	jsonParameter.add_string('original_file', "None")

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'standardscaler')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'standardscaler')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs creator
	creator = Creator(final_path, description=new_description)

	# Executing
	# Get number of traces
	n_traces = len(trs_file)
	sc = StandardScaler()
	for i in trange(n_traces, desc='[INFO *StandardScalerSK*]: Computing partial fit'):
		# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
		sc.partial_fit(np.array(trs_file[i]).reshape(1, -1))

	for i in trange(n_traces, desc='[INFO *StandardScalerSK*]: Computing standard scaler'):
		creator.Feed_Traces(sc.transform(np.array(trs_file[i]).reshape(1, -1))[0], np.frombuffer(trs_file[i].data, dtype='uint8'))
	
	print('[INFO *StandardScalerSK*]: Closing file')
	creator.Close_File()

	print('[INFO *StandardScalerSK*]: Done')
	return final_path
##################################################################################
def MinMax_Scaler_SK(trs_file, trs_mean_file=None, trs_std_file=None):

	# Get number of traces
	n_traces = len(trs_file)

	# Getting path to create new one
	original_path = trs_file.engine.path
	fname = os.path.splitext(os.path.basename(original_path))[0]
	path = os.path.dirname(os.path.abspath(original_path))
	final_path = os.path.join(path, '{}-skminmaxscaler.trs'.format(fname))

	# Create json parameters
	jsonParameter = sutils.JSONManager()
	jsonParameter.add_string('trs_mean_file', '')
	jsonParameter.add_string('original_file', '')

	# Get previous description and compose new one
	jsonManager = sutils.JSONManager.From_String(trs_file.get_header(Header.DESCRIPTION))
	new_description = trs_file.get_header(Header.DESCRIPTION)
	if jsonManager is not None:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', sutils.get_last_id_process(jsonManager) + 1)
		jsonProcess.add_string('name', 'minmaxscaler')
		jsonProcess.add_string('description', 'no desc')
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

	else:
		jsonProcess = sutils.JSONManager()
		jsonProcess.add_string('id', 1)
		jsonProcess.add_string('name', 'minmaxscaler')
		jsonProcess.add_string('description', new_description)
		jsonProcess.add_jsonmanager('parameters', jsonParameter)

		jsonManager = sutils.JSONManager()
		jsonManager.add_array('process', [])
		jsonManager.append_to_array('process', jsonProcess.JSONObject)
		jsonManager.add_string('data_dtype', 'uint8')
		new_description = jsonManager.to_string()

	# Define trs creator
	creator = Creator(final_path, description=new_description)

	# Executing
	mms = MinMaxScaler()
	for i in trange(n_traces, desc='[INFO *StandardScalerSK*]: Computing partial fit'):
		# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
		mms.partial_fit(np.array(trs_file[i]).reshape(1, -1))

	for i in trange(n_traces, desc='[INFO *StandardScalerSK*]: Computing standard scaler'):
		creator.Feed_Traces(mms.transform(np.array(trs_file[i]).reshape(1, -1))[0], np.frombuffer(trs_file[i].data, dtype='uint8'))
	
	print('[INFO *StandardScalerSK*]: Closing file')
	creator.Close_File()

	print('[INFO *StandardScalerSK*]: Done')
	return final_path
##################################################################################
