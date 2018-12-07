from torch.multiprocessing import Process
import random

class SampleCreator(Process):
    """
      Responsible for starting multile reservoir samples across
      the memory hierarchy

      Design reference: https://stackoverflow.com/questions/17172878/using-pythons-multiprocessing-process-class
    """

    def __init__(self, data_store, event):
        super(SampleCreator, self).__init__()
        self.ds = data_store
        self.num_train_points = self.ds.num_train_points
        self.sample_size = self.ds.sample_size
        self.stop_sample_creator = event

    def run(self):
        """
          Keep creating samples in the background.
        """
        while not self.stop_sample_creator.is_set():
            if self.ds.samples.full():
                continue
            else:
                self.create_sample()

    def create_sample(self):
        points = []
        i = 0
        while i < self.sample_size and i < self.num_train_points:
            points.append(i)
            i += 1

        while i >= self.sample_size and i < self.num_train_points:
            p = random.randint(0, i)
            if p < self.sample_size:
                points[p] = i
            i += 1

        reservoir = self.ds.build_reservoir_sample(points)
        self.ds.samples.put(reservoir)
