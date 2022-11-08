from sutils.directory import create_new_train_dir
from .singleton import SingletonMeta
import os

"""
[28/10/2021 16:29] 
28/10/2021 16:29 Reunión iniciada:

[28/10/2021 16:32] 
Ileana Buhan (Guest) (Invitado) se ha unido temporalmente al chat.

[28/10/2021 16:35] Ileana Buhan (Guest)
is evrything ok?

[28/10/2021 16:36] 
Ileana Buhan (Guest) (Invitado) se ha unido temporalmente al chat.

[28/10/2021 16:39] 
Ileana Buhan (Guest) (Invitado) ya no tiene acceso al chat.

[28/10/2021 16:54] Ileana Buhan (Guest)
https://eprint.iacr.org/2020/058.pdf

[28/10/2021 17:02] Ileana Buhan (Guest)
https://www.dac.com/DAC-2022/2022-Call-for-Contributions/Research-Manuscript-Submissions

[28/10/2021 17:22] 
Ileana Buhan (Guest) (Invitado) ya no tiene acceso al chat.




[12:33] Lejla Batina (Guest)
https://www.dac.com/DAC-2022/2022-Call-for-Contributions/Research-Paper-Submission-Categories

[12:41] 
Lejla Batina (Guest) (Invitado) ya no tiene acceso al chat.

[12:41] 
Ileana Buhan (Guest) (Invitado) ya no tiene acceso al chat.


"""






class SCAProject(metaclass=SingletonMeta):

    def __init__(self, project_name:str, root_path:str, id_train:str=None, method_path_destination:str=None) -> None:    
        self.TrainID:str                = id_train
        self.MethodPathDestination:str  = method_path_destination
        self.ProjectRootPath:str        = root_path
        self.__m_dataset_path_dict:dict = {}
        self.__m_train_dir              = None
        self.__m_project_name           = project_name

    def add_dataset_path(self, dataset_path:str, name:str=None):
        self.__m_dataset_path_dict[name if name is not None else len(self.__m_dataset_path_dict.keys())] = dataset_path
    
    def get_dataset_path(self, name:str=None):
        if name in self.__m_dataset_path_dict.keys():
            return self.__m_dataset_path_dict[name]
        elif name is None and len(self.__m_dataset_path_dict.keys()) > 0:
            return list(self.__m_dataset_path_dict.values())[0]
        return None
    
    @property
    def TrainDir(self):
        return self.__m_train_dir

    def update(self):
        if self.TrainID is None:
            self.__m_train_dir = create_new_train_dir(os.path.join(self.ProjectRootPath), False)
            self.TrainID = self.__m_train_dir.split(os.sep)[-1]
        else:
            self.__m_train_dir = os.path.join(self.ProjectRootPath, self.TrainID)

        if not os.path.exists(os.path.join(self.__m_train_dir, self.__m_project_name)):
            os.mkdir(os.path.join(self.__m_train_dir, self.__m_project_name))
        method_path = os.path.join(self.__m_train_dir, self.__m_project_name)

        if self.MethodPathDestination is None:
            self.MethodPathDestination = create_new_train_dir(method_path, False)
        else:
            self.MethodPathDestination = os.path.join(method_path, self.MethodPathDestination)