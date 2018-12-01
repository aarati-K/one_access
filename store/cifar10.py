from store import DataStore
import pickle
import os
import numpy as np
from pathlib import Path
from memory_hierarchy import StorageAttributes, StorageComponents
from sample_creator import SampleCreator


class Cifar10(DataStore):
    def __init__(self, **kwargs):
        super(Cifar10, self).__init__(**kwargs)
        self.dataset_name = "Cifar-10"

    def count_num_points(self):
        self.num_points = 50000

    def generate_IR(self):
        data_file_path = self.mem_config.get(StorageComponents.HDD).get(StorageAttributes.ONE_ACCESS_DIR)+'/data/'
        if not Path(data_file_path).exists():
            os.mkdir(data_file_path)
        data_file_path += self.dataset_name + '.npy'
        with Path(data_file_path).open('ab') as f:
            for root, sub_dirs, files in os.walk(self.dataset_dir):
                for file in files:
                    if "data_batch" in file:
                        path = self.dataset_dir +'/'+ file
                        fo = open(path, 'rb')
                        p = pickle.load(fo, encoding='bytes')
                        data = np.array(p.get(b'data'))
                        labels = np.array(p.get(b'labels'))
                        data_points= []
                        for data_point, label in zip(data,labels):
                            data_points.append([data_point, label])
                        nparr = np.array(data_points)
                        np.save(f,nparr)

    def generate_samples(self):
        for i in range(self.max_samples-1):
            s = SampleCreator(self)
            s.create_samples()
