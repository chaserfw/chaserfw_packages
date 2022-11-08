from abc import ABC, abstractmethod
import enum
##################################################################################
class ASCADGroupType(enum.Enum):
	PROFILING = 1
	ATTACK = 2
	BOTH = 3
##################################################################################
class ASCADGroup(ABC):
    def __init__(self, h5_stream, group, labels, traces, meta_data_table=None):
        self.Labels = labels
        self.Traces = traces
        self.Group  = group
        self.h5Stream = h5_stream
        self.meta_data_table = meta_data_table
        self._index_dict = None
    
    @property
    def h5Stream(self):
        return self._h5_stream
    
    @h5Stream.setter
    def h5Stream(self, h5_stream):
        self._h5_stream = h5_stream
        
    @property
    def MetaDataTable(self):
        return self.meta_data_table

    @property
    def Labels(self):
        return self._labels
    
    @Labels.setter
    def Labels(self, labels):
        self._labels = labels
    
    @property
    def TotalLabels(self):
        return self._labels.shape[0]

    @property
    def Traces(self):
        return self._traces
    
    @Traces.setter
    def Traces(self, traces):
        self._traces = traces

    @property
    def Group(self):
        return self._group
    
    @Group.setter
    def Group(self, group):
        self._group = group
    
    @property
    def TracesLength(self):
        return self._traces.shape[1]
    
    @property
    def TotalTraces(self):
        return self._traces.shape[0]
    
    @property
    def Info(self):
        return ('Trace shape: {} ; Label shape: {}'.format(self._traces.shape, self._labels.shape))
    
    def close(self):
        if self.h5Stream is not None:
            self.h5Stream.close()

    @property
    def IndexDict(self):
        if self._index_dict is None:
            print ('[WARNING]: Dictionary of index is None, call generate_index_labels functions first')
        return self._index_dict
    
    def generate_index_labels(self):
        from . import get_index_labels
        self._index_dict = get_index_labels(self)
###############################################################################
class Profiling(ASCADGroup):
    def __init__(self, h5_stream=None, group=None, labels=None, traces=None, metadata=None):
        super().__init__(h5_stream, group, labels, traces, metadata)
###############################################################################
class Attack(ASCADGroup):
    def __init__(self, h5_stream=None, group=None, labels=None, traces=None, metadata=None):
        super().__init__(h5_stream, group, labels, traces, metadata)