import PIL

#TODO: Create a memory config file

class DataStore():
  self.folder_name = None
  
  def __init__(self):
    pass

class StoreCreator():
  """
    Base class for all store creators for different datasets
  """
  self.transforms = []
  self.name = ""
  self.folder_name = ""

  # Initial samples pinned to heap memory, passed by the main process
  self.initial_samples = None

  def __init__(self, transforms=[]):
    self.transforms = transforms

  def countNumPoints(self):
    pass

  def generateIR(self):
    """
      Generates multiple files with (k, v) pairs stored sequentially
      with transforms applied. Generate the metadata file.
    """
    pass

  def generateSamples(self):
    """
      Create a SampleCreator object and create multiple samples
    """
    pass

  def initialize(self):
    """
      Calls generateIR and generateSamples
    """
    pass
