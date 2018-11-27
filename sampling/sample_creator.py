from multiprocessing import Process
import random


class SampleCreator(Process):
  """
    Responsible for starting multile reservoir samples across
    the memory hierarchy
    
    Design reference: https://stackoverflow.com/questions/17172878/using-pythons-multiprocessing-process-class
  """

  def __init__(self, data_store):
    super(P, self).__init__()
    self.ds = data_store
    self.num_points = self.ds.num_points
    self.sample_size = self.ds.sample_size

  def run(self):
    """
      Keep creating samples in the background.
    """
    while True:
      i = 0
      point = 0
      points = []
      while i < self.sample_size and i < self.num_points:
        point = self.get_next_point()
        points.append((i, point))
        i += 1
        point += 1
        
      while i >= self.sample_size and i < self.num_points:
        m = self.sample_size
        s = random.randint(0, self.sample_size + i)
        if s < m:
          points.append((m, point))
        i += 1
        point += 1
      
      reservoir = build_reservoir(points, self.sample_size)

      # Blocks if max number of samples met
      # TODO: Only for memory sampling rn, add SSD/Disk support later
      self.ds.samples.put(reservoir)
    
  def build_reservoir(self, points, sample_size):
    """ Get next blob into memory, read points from it, and then move on to
      next blob from ir location. """
    reservoir = Sample(sample_size)
    # TODO: Fill reservoir
    return reservoir
