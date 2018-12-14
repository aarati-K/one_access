from torch.multiprocessing import Process
import random


class SampleCreator(Process):
    """
      Responsible for starting multiple reservoir samplers across the memory hierarchy

      Design reference: https://stackoverflow.com/questions/17172878/using-pythons-multiprocessing-process-class
    """

    def __init__(self, data_store, event=None, epochs=1, replace=False, sampled=[]):
        super(SampleCreator, self).__init__()
        self.ds = data_store
        self.num_train_points = self.ds.num_train_points
        self.sample_size = self.ds.sample_size
        self.epochs = epochs
        self.sampled = sampled
        self.replace = replace
        self.stop_sample_creator = event

    def run(self):
        """
          Keep creating samples in the background.
        """
        epochs_done = 0
        while not self.stop_sample_creator.is_set() and epochs_done < self.epochs:
            for sample_queue in self.ds.samples:
                if sample_queue.full():
                    continue
                else:
                    self.create_sample(sample_queue)
                    if len(self.sampled) == self.num_train_points:
                        epochs_done += 1
                        del self.sampled
                        self.sampled = []

    def create_sample(self, sample_queue):
        points = []
        # Tracks the index of all points
        i = 0
        # Tracks the index of all points considered for sampling
        j = 0
        while j < self.sample_size and i < self.num_train_points:
            if not self.replace and i in self.sampled:
                i += 1
                continue
            points.append(i)
            j += 1
            i += 1

        while i < self.num_train_points:
            if not self.replace and i in self.sampled:
                i += 1
                continue
            p = random.randint(0, j)
            if p < self.sample_size:
                points[p] = i
            i += 1
            j += 1

        self.sampled.extend(points)
        reservoir = self.ds.build_reservoir_sample(points)
        sample_queue.put(reservoir)
        del reservoir
