from sutils import JSONManager
from .. import SettingsCrawler
from typing import Union
from sutils import format_name

class AttackCrawler(SettingsCrawler):
	def __init__(self, json_manager:Union[JSONManager, str]=None) -> None:
		super(AttackCrawler, self).__init__(json_manager)
		
	def create_attack(self, file_name='guessing_entropy', auto_suffix=True):
		attack_list = self._json_manager.get_value('attacks_list')
		
		python_filename = '{}_{}.npy'.format(file_name, format_name() if auto_suffix else '')
		plot_filename = '{}_{}.pdf'.format(file_name, format_name() if auto_suffix else '')
		attackJSON = JSONManager()
		attackJSON.add_string('npy_file', python_filename)
		attackJSON.add_string('pdf_file', plot_filename)
		if attack_list is None:
			attack_list = []

		attack_list.append(attackJSON.JSONObject)
		self._json_manager.add_array('attacks_list', attack_list)

		return (plot_filename, python_filename)
