from sutils import JSONManager
from strainfuctions import AbstractTrainer
from dataloaders import ProfileLoader
from dataloaders import AttackLoader
from dataloaders import DataLoader
from typing import Union


class SettingsCrawler():
	def __init__(self, json_manager: Union[JSONManager, str] = None) -> None:

		if json_manager is not None:
			if isinstance(json_manager, str):
				self._json_manager = JSONManager(path_to_load=json_manager)
			else:
				self._json_manager = json_manager
		else:
			self._json_manager = JSONManager()

	def save_settings(self, path: str) -> None:
		self._json_manager.save(path)

	def from_trainer(self, trainer: AbstractTrainer) -> None:
		self._json_manager.add_string('epochs', trainer.Epochs)
		self._json_manager.add_string('batch_size', trainer.BatchSize)
		self._json_manager.add_string('max_lr', trainer.MaxLR)
		self._json_manager.add_string('ocp_sps', trainer.OCPsps)
		self._json_manager.add_string('train_dir', trainer.TrainDir)

		if trainer.XTrainDict:
			for _, value in trainer.XTrainDict.items():
				self._json_manager.add_string('train_shape', value.shape)

		if trainer.XValDict:
			for _, value in trainer.XValDict.items():
				self._json_manager.add_string('val_shape', value.shape)

	def from_dataloader(self, profile_loader: ProfileLoader = None, attack_loader: AttackLoader = None) -> None:
		if profile_loader is None and attack_loader is None:
			profile_loader = DataLoader.ProfileLoader
			attack_loader = DataLoader.AttackLoader

		if profile_loader is not None:
			profileLoadersJSON = JSONManager()
			profileLoadersJSON.add_string(
				'loader_method', profile_loader.LoaderMethod)
			profileLoadersJSON.add_string(
				'type_loader', profile_loader.TypeLoader)

			self._json_manager.add_jsonmanager(
				'profile_loader', profileLoadersJSON)

		if attack_loader is not None:
			attackLoadersJSON = JSONManager()
			attackLoadersJSON.add_string(
				'loader_method', attack_loader.LoaderMethod)
			attackLoadersJSON.add_string(
				'type_loader', attack_loader.TypeLoader)

			self._json_manager.add_jsonmanager(
				'attack_loader', attackLoadersJSON)
				
	def add_string(self, key: str, value: any) -> None:
		self._json_manager.add_string(key, value)

	def add_array(self, key: str, array: any) -> None:
		self._json_manager.add_array(key, array)

	def append_to_array(self, array_name: str, object_value: any) -> None:
		self._json_manager.append_to_array(array_name, object_value)

	def get_value(self, key) -> any:
		return self._json_manager.get_value(key)

	@property
	def JSONManager(self) -> JSONManager:
		return self._json_manager
	
	@property
	def JSONObject(self):
		return self._json_manager.JSONObject
