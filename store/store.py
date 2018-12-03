import os
import PIL
from sampling.sample import Sample
from store.memory_hierarchy import MemoryHierarchy
from torch.multiprocessing import Array


class Metadata():
    def __init__(self):
        pass


class DataStore():
    """
      Base class for all store creators for different datasets
    """
    # relative path names of train and test folders
    TRAIN_FOLDER = "/train"
    TEST_FOLDER = "/test"
    DATA_FILE = "data_{}.npy"

    def __init__(self, input_data_folder, max_batches=1, transform=None, target_transform=None, max_samples=1, sample_size=100,
                 batch_size=128, delete_existing=False):
        self.dataset_name = ""
        self.transform = transform
        self.target_transform = target_transform

        # The folder containing the input data, from which IR is generated
        self.input_data_folder = input_data_folder

        self.mem_config = None
        self.metadata = None
        self.num_train_points = 0
        self.num_test_points = 0
        self.delete_existing = delete_existing

        self.max_samples = max_samples
        self.sample_size = sample_size
        # Samples populated by the SampleCreator process (shared memory)
        self.samples = Array('f', 0)

        self.max_batches = max_batches
        self.batch_size = batch_size
        # batches populated by the BatchCreator process (shared memory)
        self.batches = Array('f', 0)

    def count_num_points(self):
        # Use this implementation for default format of subdirectory classes
        # (typically for image datasets), else override.
        # Go through the input_data_folder and count number of points
        num_train_points = 0
        for root, subdirs, files in os.walk(self.input_data_folder):
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

    def get_data_folder_path(self):
        """
            Return the folder containing the data in intermediate rep (IR)
        """
        pass

    def initialize_shared_mem(self):
        """
            Initialize the self.samples and self.batches. These are shared
            with the SampleCreator and BatchCreator processes.
        """
        # Decide on the size and data type of self.samples and self.batches
        pass

    def generate_samples(self):
        """
          Create a SampleCreator object and create multiple samples
        """
        # Decide on the number of samples to create at each level of
        # the memory hierarchy. (If there is no SSD, no need to create samples)
        # self.samples refers to the reservoir samples in memory
        pass

    def initialize(self):
        """
          Calls generateIR and generateSamples
        """
        MemoryHierarchy.load()
        self.mem_config = MemoryHierarchy.mem_config
        self.generate_IR()
        self.initialize_shared_mem()
        self.generate_samples()
