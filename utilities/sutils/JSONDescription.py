from .jsonmanager import JSONManager
from .jsonmanager import load_json_from_string

class JSONMetadataColumn(JSONManager):
	def __init__(self, name, lbyte, fbyte):
		"""
		docstring
		"""
		super(JSONMetadataColumn, self).__init__(path_to_load=None)
		self.Name = name
		self.Lbyte = lbyte
		self.Fbyte = fbyte
	
	@property
	def Name(self):
		"""
		docstring
		"""
		return self.get_value('name')

	@Name.setter
	def Name(self, name):
		"""
		docstring
		"""
		self.add_string('name', name)
	
	@property
	def Lbyte(self):
		"""
		docstring
		"""
		return self.get_value('lbyte')

	@Lbyte.setter
	def Lbyte(self, lbyte):
		"""
		docstring
		"""
		self.add_string('lbyte', lbyte)
	
	@property
	def Fbyte(self):
		"""
		docstring
		"""
		return self.get_value('fbyte')

	@Fbyte.setter
	def Fbyte(self, fbyte):
		"""
		docstring
		"""
		self.add_string('fbyte', fbyte)

class JSONDescription(JSONManager):
	def __init__(self, json_string=None):
		"""
		docstring
		"""
		super(JSONDescription, self).__init__(path_to_load=None)
		if json_string is not None:
			self._json_object = JSONManager.From_String(json_string)._json_object
	
	def Set_Mdata_Definition(self, list_metadataColumns):
		#if not super.has_key(self, 'mdata_meta_definition'):
		super.add_array('mdata_meta_definition', list_metadataColumns)
	
	def Get_Mdata_Definition(self):
		#if not super.has_key(self, 'mdata_meta_definition'):
		return super.get_value('mdata_meta_definition')
	
