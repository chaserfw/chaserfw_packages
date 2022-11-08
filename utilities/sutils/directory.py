import os
import sys
import tables
import numpy as np
import pickle

import subprocess
import pkg_resources
#======================================================================================
def now(clean=False):
	from datetime import datetime
	value = str(datetime.now().date())
	if clean:
		value = value.replace(':', '-')
		value = value.replace('.', '-')
	return value
#======================================================================================
def current_time(clean=False):
	from datetime import datetime
	value = str(datetime.now().time())
	if clean:
		value = value.replace(':', '-')
		value = value.replace('.', '-')
	return value
#======================================================================================
def get_file_suffix():
	from datetime import datetime
	suffix = 'D{}=H{}'.format(now(), current_time())
	suffix = suffix.replace(':', '-')
	suffix = suffix.replace('.', '-')

	return suffix
#======================================================================================
def file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if not os.path.exists(file_path):
		#print("[ERROR]: Provided file path '%s' does not exist!" % file_path)
		return False
	return True
#======================================================================================
def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if not os.path.exists(file_path):
		print("[ERROR]: Provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return
#======================================================================================
def format_name():
	"""Funcion auxiliar para create_new_train_dir
	"""
	import datetime
	currentDT = datetime.datetime.now()
	return currentDT.strftime("%Y-%m-%d %H:%M:%S").replace('-', '').replace(':', '').replace(' ', '-')
#======================================================================================
def create_new_train_dir(root, ismul=True):
	"""Esta funcion crea un directorio cuyo nombre esta compuesto por la fecha, y el timestamp
	sirve para llevar orden en cada intento de entrenamiento, con un burdo control de tiempo
	"""
	import os
	suffix = os.path.join(root, format_name() +  ('-mul' if ismul else ''))
	if not os.path.exists(suffix):
		os.mkdir(suffix)
	return suffix
#======================================================================================
def load_ascad_profiling(ascad_database_file, trace_type, load_metadata=False):
	"""Carga las trazas y los labels (ademas de la metadata) desde el archivo .h5
	que es el formato usado por ASCAD.
	ADVERTENCIA: especial cuidado con el tipo de dato en trace_type, porque podrian cargarse todo como ceros.
	"""
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = tables.open_file(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
		
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=trace_type)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.uint8)
			
	if load_metadata == False:
		return (X_profiling, Y_profiling)
	else:
		return (X_profiling, Y_profiling, in_file['Attack_traces/metadata'])

#======================================================================================
def load_ascad_attack(ascad_database_file, trace_type, load_metadata=False):
	"""Carga las trazas y los labels (ademas de la metadata) desde el archivo .h5
	que es el formato usado por ASCAD.
	ADVERTENCIA: especial cuidado con el tipo de dato en trace_type, porque podrian cargarse todo como ceros.
	"""
	
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = tables.open_file(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
		   
	# Load attacking traces
	X_attack = np.array(in_file.root.Attack_traces.traces, dtype=trace_type)
	# Load attacking labels
	Y_attack = np.array(in_file.root.Attack_traces.labels, dtype=np.uint8)
		
	if load_metadata == False:
		return (X_attack, Y_attack)
	else:
		return (X_attack, Y_attack, in_file.root.Attack_traces.metadata)
#======================================================================================
def load_ascad_attack_groups(ascad_database_file, trace_type, load_metadata=False):
	"""Carga las trazas y los labels (ademas de la metadata) desde el archivo .h5
	que es el formato usado por ASCAD.
	ADVERTENCIA: especial cuidado con el tipo de dato en trace_type, porque podrian cargarse todo como ceros.
	"""
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = tables.open_file(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
		   
	# Load attacking traces
	X_attack = in_file.root.Attack_traces.traces
	# Load attacking labels
	Y_attack = in_file.root.Attack_traces.labels
		
	if load_metadata == False:
		return (X_attack, Y_attack)
	else:
		return (X_attack, Y_attack, in_file.root.Attack_traces.metadata)
#======================================================================================
def save_scaler(path:str, scaler):
	"""Saves a scaler in the given path
	
	Args:
		path (str): Path of the directory to save the scaler
		scaler: a sklear scaler
	Returns:
		None
	"""
	f = open(path, 'wb')
	f.write(pickle.dumps(scaler))
	f.close()
#======================================================================================
def load_scaler(path):
	"""Loads a scaler from the given path
	
	Args:
		path (str): Path of the directory to save the scaler
	Returns:
		scaler: A sklear scaler
	"""
	scaler = None
	with open(path, 'rb') as f:
		scaler = pickle.load(f)
	return scaler
#======================================================================================
def check_for_modules(dict_names: dict):
	"""Checks whether the given modules are instaled, if not a set of 
	the missing package are returned

	Args:
		dict_names(dict): a dictionary with the module names

	Returns:
		missing modules (set): a set with the missing modules
	"""
	required = dict_names
	installed = {pkg.key for pkg in pkg_resources.working_set}
	missing = required - installed
	return missing


"""
def save_scaler_list(scaler_list):
	for i, scaler in enumerate(scaler_list):
		scaler_dir_names.append(os.path.join(train_dir, 'scaler_{}_{}'.format(i, train_dir.split('/')[-1])))
		f = open(scaler_dir_names[i], 'wb')
		f.write(pickle.dumps(scaler))
		f.close()
"""