from datasets.VeidDataset import VeidDataset
from datasets.RapDataset import RapDataset
from datasets.VehicleIDDataset import VehicleIDDataset



class DatasetFactory:
    def __init__(self, dataset_name, data_directory, augment=True, num_classes=None):
        self._data_directory = data_directory
        self._dataset_name = dataset_name
        self._augment = augment
        self._num_classes = num_classes


    def get_dataset(self, dataset_part):
        if self._dataset_name == 'Veid':
            return VeidDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)
        if self._dataset_name == 'Veid_view':
            return RapDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)
        if self._dataset_name == 'VehicleID':
            return VehicleIDDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)
        else:
            raise ValueError('Unknown dataset name: %s' % self._data_directory)


    def get_dataset_name(self):
        return self._dataset_name
