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
    self.md = self.data_store.metadata

  def run(self):
    """
      Keep creating samples in the background.
    """
    while True:
      reservoir = Sample(self.ds.sample_size)

      i = 0
      while i < reservoir.maxsize and i < self.md.num_points:
        point = self.get_next_point()
        reservoir.items.append(point)
        i += 1
        
      while i >= reservoir.maxsize and i < self.md.num_points:
        point = self.get_next_point()
        s = random.randint(0, reservoir.maxsize + i)
        if s < m:
          reservoir.items[s] = point
        i += 1

      # Blocks if max number of samples met
      self.ds.samples.put(reservoir)
    
  def get_next_point(self):
    """ Get next blob into memory, read points from it, and then move on to
      next blob from ir location. """
    # TODO:
    pass
