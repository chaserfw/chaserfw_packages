import trsfile
from trsfile import Header

def get_description(trs_file_path):
	trs_file = trsfile.open(trs_file_path, mode='r')
	description = trs_file.get_header(Header.DESCRIPTION)
	trs_file.close()
	return description