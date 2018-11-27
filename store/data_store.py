import PIL
from sampling.sample import Sample
from multiprocessing import Queue


class DataStore():
  """
    Base class for all store creators for different datasets
  """

  def __init__(self,
      dataset_dir,
      transforms=[],
      download=False,
      max_samples=1,
      sample_size=100
    ):
    self.transforms = transforms
    self.transforms = []
    
    self.dataset_name = ""
    self.dataset_dir = ""
    self.metadata_filepath = ""
    self.metadata = None
    self.num_points = 0

    # Initial samples pinned to heap memory, passed by the main process
    self.samples = Queue(max_samples)
    self.sample_size = sample_size

    # To be populated by the batch creator
    self.batches = None
    self.max_batches = 0

  def count_num_points(self):
    # Go through the dataset_dir and count number of points
    # Write num_points to the metadata file (in generateIR)
    pass

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
    pass

  def initialize(self):
    """
      Calls generateIR and generateSamples
    """
    pass
