import tables
from sutils import file_exists

def load_ascad_handler(origin_path:str, mode='r', force_mode=False):
	h5_stream   = None
	original_path = None
	if origin_path in tables.file._open_files.filenames:
		h5_stream = tables.file._open_files.get_handlers_by_name(origin_path)
		h5_stream = next(iter(h5_stream))
		if h5_stream.mode != mode and force_mode:
			h5_stream.close()
			h5_stream = tables.open_file(origin_path, mode=mode)
	else:
		if (file_exists(origin_path) and mode == 'r') or (not file_exists(origin_path) and mode == 'w'):
			h5_stream = tables.open_file(origin_path, mode=mode)

	original_path = origin_path
	return (h5_stream, original_path)

def get_open_tables():
	return tables.file._open_files.filenames

def force_close(filename, check=False):
	if filename in tables.file._open_files.filenames:
		h5_stream = tables.file._open_files.get_handlers_by_name(filename)
		h5_stream = next(iter(h5_stream))
		h5_stream.close()
	if check:
		print (get_open_tables())

def file_type(path_file):
	import os
	return os.path.splitext(os.path.basename(path_file))[1]

def __h5_group_crawler(node, tree_list):
	tree_list.append(str(node))
	if 'Array' not in str(node) and 'Table' not in str(node):
		for subnode in node:
			__h5_group_crawler (subnode, tree_list)

def h5_group_descriptor(h5_path_file):
	h5file = load_ascad_handler(h5_path_file)[0]
	tree_list = []
	for node in h5file.list_nodes('/'):
		__h5_group_crawler(node, tree_list)
	
	h5file.close()
	return tree_list