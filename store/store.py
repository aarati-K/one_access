import json
import os
from store.memory_hierarchy import MemoryHierarchy
from torch.multiprocessing import Queue
import numpy as np
import torch
from sampling.sample_creator import SampleCreator


class MetadataField():
    KEY_SIZE = "key_size"
    VALUE_SIZE = "value_size"
    FILES = "files"
    KV_COUNT = "kv_count"
    # Number of numpy file chunks
    CHUNK_COUNT = "chunk_count"


class Metadata():
    METADATA_FILE = "metadata.json"

    def __init__(self, data_store):
        self.data_store = data_store

    def load(self):
        """
            Recursively load the metadata from all the subfolders. Return a
            dict with keys as relative paths from
            data_store.get_data_folder_path() and value as metadata dict.
        """
        metadata_dict = {}
        data_folder = self.data_store.get_data_folder_path() + '/'
        for root, sub_folders, files in os.walk(data_folder):
            for sub_folder in sub_folders:
                metadata_dict[sub_folder] = \
                    self._load(data_folder + sub_folder)

        return metadata_dict

    def _load(self, folder_name):
        """
            Load the metadata for the specific folder. Return a dict containing
            the metadata.
        """
        for root, sub_folders, files in os.walk(folder_name):
            if self.METADATA_FILE not in files:
                return {}
            metadata_file = folder_name + '/' + self.METADATA_FILE
            metadata = open(metadata_file).read()
            return json.loads(metadata)

    def store(self, metadata_dict):
        sub_folders = metadata_dict.keys()
        data_folder = self.data_store.get_data_folder_path() + '/'
        for sub_folder in sub_folders:
            filename = data_folder + sub_folder + '/' + self.METADATA_FILE
            file = open(filename, 'w')
            file.write(json.dumps(metadata_dict.get(sub_folder)))
            file.close()

class DataStore():
    """
      Base class for all store creators for different datasets
    """
    # relative path names of train and test folders
    TRAIN_FOLDER = "train"
    TEST_FOLDER = "test"
    DATA_FILE = "data_{}.npy"

    def __init__(self, input_data_folder, max_batches=1, batch_size=1, max_samples=1,
        rel_sample_size=10, transform=None, target_transform=None, delete_existing=False):
        # To be assigned by the derived class
        self.dataset_name = ""

        # transform function to be applied to values and labels
        self.transform = transform
        self.target_transform = target_transform

        # The folder containing the input data, from which IR is generated
        self.input_data_folder = input_data_folder

        # Initialize mem_config
        MemoryHierarchy.load()
        self.mem_config = MemoryHierarchy.mem_config

        # Initialize metadata, might be uninitialized if the datastore has not
        # yet been created
        self.metadata = Metadata(self).load()

        # Statistics, computed by the derived class
        self.num_train_points = 0
        self.num_test_points = 0
        self.key_size = 0
        self.value_size = 0

        # Delete any existing IR data folder
        self.delete_existing = delete_existing

        # SAMPLING ATTRIBUTES
        self.max_samples = max_samples
        self.rel_sample_size = rel_sample_size
        self.sample_size = batch_size*rel_sample_size
        # Samples populated by the SampleCreator process (shared memory)
        self.samples = [Queue(1) for i in range(self.max_samples)]
        # Record the points sampled in advance, before starting batch creator and sample creator processes
        self.points_sampled = []

        # BATCHING ATTRIBUTES
        self.max_batches = max_batches
        self.batch_size = batch_size
        # batches populated by the BatchCreator process (shared memory)
        self.batches = Queue(self.max_batches)

    def count_num_points(self):
        # Use this implementation for default format of subfolder classes
        # (typically for image datasets), else override.
        # Go through the input_data_folder and count number of points
        num_train_points = 0
        for root, subfolders, files in os.walk(self.input_data_folder):
            for file in files:
                num_train_points += 1
        self.num_train_points = num_train_points
        # TODO: Add logic for counting num_test_points

    def generate_IR(self):
        """
          Generates multiple files with (k, v) pairs stored sequentially
          with transforms applied. Generate the metadata file.
        """
        # NOTE: Assuming values of equal size
        # Decide on key size
        # Decide on a fixed value size (after applying transforms)
        # Decide on number of files
        # Store the file with contiguous <K, V> pairs
        # Create metadata file
        # - Specify key size, value size
        # - Specify file names, and how many <K, V> pairs each has
        # - Set self.metadata field
        pass

    def transform_point(self, value):
        """
            To be defined by the derived class
        """
        return value[0], value[1]

    def get_data_folder_path(self):
        """
            Return the folder containing the data in intermediate rep (IR)
        """
        pass

    def build_reservoir_sample(self, points):
        """
            Build a reservoir sample with points
        """
        points.sort()
        data = []
        labels = []
        file_to_points_map = {}

        cur_data_file_index = 0
        cur_data_file_path = self.DATA_FILE.format(str(cur_data_file_index))
        metadata = self.metadata.get(self.TRAIN_FOLDER)
        cur_file_metadata = metadata[MetadataField.FILES][cur_data_file_path]
        cur_file_min_index = 0
        cur_file_max_index = cur_file_metadata[MetadataField.KV_COUNT] # - 1

        points_from_cur_file = []
        for point in points:
            if point >= cur_file_max_index:
                # Save the cur files points to the map
                points_in_file = map(lambda point: point-cur_file_min_index, points_from_cur_file)
                file_to_points_map[cur_data_file_path] = points_in_file

                # Find the next file
                points_from_cur_file = []
                while point >= cur_file_max_index:
                    cur_data_file_index += 1
                    cur_data_file_path = self.DATA_FILE.format(str(cur_data_file_index))
                    cur_file_metadata = metadata[MetadataField.FILES][cur_data_file_path]
                    cur_file_min_index = cur_file_max_index
                    cur_file_max_index += cur_file_metadata[MetadataField.KV_COUNT]

            # get points from current file
            points_from_cur_file.append(point)

        # Entry for the last file
        points_in_file = map(lambda point: point-cur_file_min_index, points_from_cur_file)
        file_to_points_map[cur_data_file_path] = points_in_file
        # Iterate over all the files
        for filename in file_to_points_map.keys():
            points_in_file = file_to_points_map[filename]
            self.get_sample_from_file(filename, points_in_file, data, labels)

        return [np.array(data), np.array(labels)]

    def get_sample_from_file(self, filename, points, data, labels):
        if not points:
            return

        full_path = self.get_data_folder_path() + '/' + self.TRAIN_FOLDER \
            + '/' + filename
        metadata = self.metadata[self.TRAIN_FOLDER][MetadataField.FILES][filename]
        num_chunks = metadata[MetadataField.CHUNK_COUNT]
        chunks_read = 1
        f = open(full_path, 'rb')
        cur_chunk = np.load(f)
        cur_chunk_min_index = 0
        cur_chunk_max_index = cur_chunk.shape[0]

        for point in points:
            while point >= cur_chunk_max_index:
                chunks_read += 1
                if chunks_read > num_chunks:
                    print("Did not find chunk for index ", point)
                cur_chunk = np.load(f)
                cur_chunk_min_index = cur_chunk_max_index
                cur_chunk_max_index += cur_chunk.shape[0]

            value = cur_chunk[point-cur_chunk_min_index]
            d, l = self.transform_point(value)
            data.append(d)
            labels.append(l)

    def initialize_samples(self):
        """
          Create a SampleCreator object and create multiple samples
        """
        # Decide on the number of samples to create at each level of
        # the memory hierarchy. (If there is no SSD, no need to create samples)
        # self.samples refers to the reservoir samples in memory
        s = SampleCreator(self)
        for sample_queue in self.samples:
            s.create_sample(sample_queue)
        # Record the points already sampled
        # NOTE: performing sampling without replacement by default
        self.points_sampled = s.sampled

    def initialize(self):
        """
          Calls generateIR and generateSamples
        """
        self.generate_IR()
        self.count_num_points()
        self.initialize_samples()
