from store.store import DataStore, Metadata, MetadataField
import pickle
import os
import numpy as np
from pathlib import Path
from store.memory_hierarchy import StorageAttributes, StorageComponents
from sampling.sample_creator import SampleCreator


class Cifar10(DataStore):
    def __init__(self, **kwargs):
        super(Cifar10, self).__init__(**kwargs)
        self.dataset_name = "Cifar-10"
        self.metadata = Metadata(self).load()
        train_metadata = self.metadata.get(self.TRAIN_FOLDER)
        if train_metadata:
            self.key_size = train_metadata.get(MetadataField.KEY_SIZE)
            self.value_size = train_metadata.get(MetadataField.VALUE_SIZE)
        else:
            self.key_size = self.value_size = None

    def count_num_points(self):
        self.num_train_points = 50000
        self.num_test_points = 10000

    def generate_IR(self):
        data_folder_path = self.get_data_folder_path()
        if not Path(data_folder_path).exists():
            print("Creating directory(s)", data_folder_path)
            Path(data_folder_path).mkdir(parents=True, exist_ok=True)

        # Create train and test directories
        train_folder_path = data_folder_path + '/' + self.TRAIN_FOLDER
        test_folder_path = data_folder_path + '/' + self.TEST_FOLDER
        for path in [train_folder_path, test_folder_path]:
            if not Path(path).exists():
                print("Creating directory", path)
                Path(path).mkdir(parents=True, exist_ok=True)

        # Create the train data file
        train_file_path = train_folder_path + '/' + self.DATA_FILE.format(0)
        if Path(train_file_path).exists():
            msg = "File " + train_file_path + "exists. "
            if self.delete_existing:
                msg += "Deleting it."
                print(msg)
                os.remove(train_file_path)
            else:
                msg += "Skipping."
                print(msg)
                return

        # Generate train dataset
        f_train = Path(train_file_path).open('ab')
        for root, sub_dirs, files in os.walk(self.input_data_folder):
            for file in files:
                if "data_batch" not in file:
                    continue
                full_path = self.input_data_folder +'/'+ file
                nparr = self.read_cifar_batch(full_path)
                np.save(f_train, nparr)

        # Generate test dataset
        test_file_path = test_folder_path + '/' + self.DATA_FILE.format(0)
        f_test = Path(test_file_path).open('ab')
        test_data_file = self.input_data_folder + "/test_batch"
        nparr = self.read_cifar_batch(test_data_file)
        np.save(f_test, nparr)

        # Create metadata for train and test folders
        self.write_metadata()

    def read_cifar_batch(self, batch_filename):
        f = open(batch_filename, 'rb')
        p = pickle.load(f, encoding='bytes')
        data = np.array(p.get(b'data'))
        labels = np.array(p.get(b'labels'))
        data_points= []
        for data_point, label in zip(data,labels):
            data_points.append([data_point, label])

        # Assign key and value size, if not already assigned
        if not self.key_size:
            self.key_size = 4 # bytes
            # value size + 1 (for label)
            self.value_size = (len(data_points[0][0]) + 1)*4 #bytes

        return np.array(data_points)

    def write_metadata(self):
        metadata_dict = {}
        train_metadata = {
            MetadataField.KEY_SIZE: self.key_size,
            MetadataField.VALUE_SIZE: self.value_size,
            MetadataField.FILES: {
                self.DATA_FILE.format('0'): {
                    MetadataField.KV_COUNT: 50000,
                    MetadataField.CHUNK_COUNT: 5,
                }
            }
        }
        metadata_dict[self.TRAIN_FOLDER] = train_metadata

        test_metadata = {
            MetadataField.KEY_SIZE: self.key_size,
            MetadataField.VALUE_SIZE: self.value_size,
            MetadataField.FILES: {
                self.DATA_FILE.format('0'): {
                    MetadataField.KV_COUNT: 10000,
                    MetadataField.CHUNK_COUNT: 1,
                }
            }
        }
        metadata_dict[self.TEST_FOLDER] = test_metadata

        metadata = Metadata(self)
        metadata.store(metadata_dict)
        self.metadata = metadata.load()

    def get_data_folder_path(self):
        return self.mem_config.get(StorageComponents.HDD)\
            .get(StorageAttributes.ONE_ACCESS_DIR) + '/' + self.dataset_name

    def generate_samples(self):
        for i in range(self.max_samples-1):
            s = SampleCreator(self)
            s.create_samples()
