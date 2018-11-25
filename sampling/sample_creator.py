class SampleCreator():
  """
    Responsible for starting multile reservoir samples across
    the memory hierarchy
  """
  self.data_store = None
  self.target_location = ""

  def __init__(self):
    pass

  def create_samples(self):
    """
      Keep creating samples in the background.
    """
    # Generate keys using reservoir sampling with num_points in 
    # the metadata file.
    # Read the keys from one level below the hierarchy,
    # and save the samples to the target location.
    pass