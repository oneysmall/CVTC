import os
import tensorflow as tf
from datasets.Dataset import Dataset


RAP_MEAN = [107.460081, 107.206399, 108.914598]
RAP_STD = [54.710137, 54.548842, 54.995074]

class RapDataset(Dataset):
    FILE_BY_PART = {'train': 'new.txt', 'test': 'test_list.txt', 'val': 'val_list.txt'}
    CROP_BORDER = 0.05
    
    def __init__(self, data_directory, dataset_part, augment=True, num_classes=None):
        if num_classes is None:
            #num_classes = 51
            num_classes=576         
        super().__init__(mean=RAP_MEAN, std=RAP_STD, num_classes=num_classes, data_directory=data_directory, dataset_part=dataset_part, augment=augment)

    def get_input_data(self, is_training):
        data_file_name = self.get_data_file_for_mode()
        
        file_name_list = []
        paths_list = []
        labels_list = []
        views_list = []
        color_list = []
        type_list = []
        
        with open(data_file_name, 'r') as reader:
            for line in reader.readlines():
                space_split = line.split(' ')
                
                file_name = space_split[0]
                labels = int(file_name[6:10])-1
                view = int(space_split[1])-1 
                colors = int(file_name[28:30])-1
                types = int(file_name[32])-1
                file_name_list.append(file_name)
                paths_list.append(os.path.join(self._data_directory, file_name))
                labels_list.append(labels)
                views_list.append(view)
                color_list.append(colors)
                type_list.append(types)

        label_mapping = {label: index for index, label in enumerate(list(sorted(set(labels_list))))}
        labels = [label_mapping[actual_label] for actual_label in labels_list]

        print('Read %d image paths for processing from %s' % (len(file_name_list), data_file_name))
        return file_name_list, paths_list, labels, views_list, color_list, type_list

    def get_number_of_samples(self):
        data_file = self.get_data_file_for_mode()
        
        with open(data_file, 'r') as reader:
            return len(reader.readlines())
    
    def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
        file_name_tensor, image_path_tensor, label_tensor, view_tensor, color_tensor, type_tensor = sliced_input_data
        image_tensor = self.read_and_distort_image(file_name_tensor, image_path_tensor, image_size)
        return self.get_dict_for_batching(file_name_tensor=file_name_tensor, image_path_tensor=image_path_tensor, multi_class_label=label_tensor, image_tensor=image_tensor, view_label=view_tensor, color_tensor=color_tensor, type_tensor=type_tensor)
    
    def get_input_function_dictionaries(self, batched_input_data):
        return {'file_names': batched_input_data['file_name'], 'paths': batched_input_data['path'], 'images': batched_input_data['image']}, \
            {'multi_class_label': batched_input_data['multi_class_label'], 'views': batched_input_data['view'], 'colors':batched_input_data['color'], 'types':batched_input_data['type'],'labels':batched_input_data['multi_class_label']}
    
    def get_data_file_for_mode(self):
        data_file = self.FILE_BY_PART[self._dataset_part]
        return os.path.join(self._data_directory, data_file)
