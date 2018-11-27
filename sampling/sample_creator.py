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

  def run(self):
    """
      Keep creating samples in the background.
    """
    while True:
      reservoir = Sample(self.ds.sample_size)

      i = 0
      while i < reservoir.maxsize and i < self.num_points:
        point = self.get_next_point()
        reservoir.items.append(point)
        i += 1
        
      point = reservoir.maxsize
      points_to_fill = []
      while i >= reservoir.maxsize and i < self.num_points:
        point += 1
        s = random.randint(0, reservoir.maxsize + i)
        if s < m:
          points_to_fill.append((m, point))
        i += 1
      
      fill_reservoir_with_points(reservoir, points_to_fill)

      # Blocks if max number of samples met
      # TODO: Only for memory sampling rn, add SSD/Disk support later
      self.ds.samples.put(reservoir)
    
  def fill_reservoir_with_points(self, reservoir, points):
    """ Get next blob into memory, read points from it, and then move on to
      next blob from ir location. """
    # TODO:
    pass
