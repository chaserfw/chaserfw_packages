"""
@author: Servio Paguada
@email: serviopaguada@gmail.com
"""
import json

#======================================================================================
def numpy2json(path_name, numpy_array):
	import numpy as np
	import codecs
	import json 
	
	listed_numpy_array = numpy_array.tolist()
	file_path = path_name
	json.dump(listed_numpy_array, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
#======================================================================================
def save_json(path_name, json_string):
	import codecs
	import json
	file_path = path_name
	json.dump(json_string, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
#======================================================================================
def classes_indexes2json(path_name, classes_indexes):
	import codecs
	import json
	file_path = path_name
	json.dump(classes_indexes, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
#======================================================================================
def load_json(path_name):
	import json
	jsonObject = None
	with open(path_name) as json_file:
		jsonObject = json.load(json_file)
	return jsonObject
#======================================================================================
def load_json_from_string(string):
	jsonObject = None
	try:
		jsonObject = json.loads(string)
	except:
		print('[INFO *JSONManager*]: Not possible to create json from string, returning None')
	
	return jsonObject
#======================================================================================
def get_last_id_process(jsonmanager):
	if not jsonmanager.has_key('process'):
		return 0
	else:
		process_list = jsonmanager.get_value('process')
		return len(process_list)
#======================================================================================

class JSONManager():
	def __init__(self, path_to_load=None):
		if path_to_load is not None:
			self.load(path_to_load)
		else:
			self._json_object = {}

	@staticmethod
	def From_String(string):
		jsonManager = JSONManager()
		jsonManager._json_object = load_json_from_string(string)
		if jsonManager._json_object is None:
			return None
		else:
			return jsonManager

	@property
	def JSONObject(self):
		return self._json_object

	def add_array(self, key, array):
		self._json_object[key] = array

	def add_string(self, key, stri):
		self._json_object[key] = stri

	def add_jsonmanager(self, key, jsonmanager):
		self._json_object[key] = jsonmanager.JSONObject

	def append_to_array(self, array_name, object_value):
		sub_object = self.JSONObject[array_name]
		sub_object.append(object_value)
	
	def get_value(self, key):
		value = None
		if key in self._json_object:
			value = self._json_object[key]
		return value
	
	def has_key(self, key):
		return (key in self._json_object)

	def save(self, path_name=None):
		if path_name is None:
			save_json(self.__path_to_load, self.JSONObject)
		else:
			save_json(path_name, self.JSONObject)
	
	def load(self, path_name):
		self._json_object = load_json(path_name)
		
	def setJSONobject(self, dic):
		if dic is None:
			self._json_object = {}
		else:	
			self._json_object = dic

	def to_string(self):
		return json.dumps(self._json_object)
#======================================================================================