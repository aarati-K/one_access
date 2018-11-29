import os
import PIL
from .memory_hierarchy import MemoryHierarchy
from sampling.sample import Sample
from torch.multiprocessing import Process


class Metadata():
    def __init__(self):
        pass

class DataStore():
    """
      Base class for all store creators for different datasets
    """

    def __init__(self, dataset_dir, max_batches, transforms=[], download=False, max_samples=1, sample_size=100):
        self.transforms = transforms
        self.transforms = []

        self.dataset_name = ""
        self.mem_config = None
        self.dataset_dir = dataset_dir
        self.metadata_filepath = ""
        self.metadata = None
        self.num_points = 0

        # Initial samples pinned to heap memory, passed by the main process
        self.samples = Queue(max_samples)
        self.sample_size = sample_size

        # To be populated by the batch creator
        self.batches = None
        self.max_batches = max_batches

    def count_num_points(self):
        # Go through the dataset_dir and count number of points
        # Write num_points to the metadata file (in generateIR)
        num_points = 0
        for root, subdirs, files in os.walk(self.dataset_dir):
            for file in files:
                num_points += 1
        self.num_points = num_points

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

    def generate_samples(self):
        """
          Create a SampleCreator object and create multiple samples
        """
        # Decide on the number of samples to create at each level of
        # the memory hierarchy. (If there is no SSD, no need to create samples)
        # self.samples refers to the reservoir samples in memory

    def initialize(self):
        """
          Calls generateIR and generateSamples
        """
        MemoryHierarchy.load()
        self.mem_config = MemoryHierarchy.get_config()
        generate_IR()
        generate_samples()
