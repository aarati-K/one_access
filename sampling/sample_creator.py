from torch.multiprocessing import Process
import random
import time


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
        if len(sampled):
            self.sampled = sampled
        else:
            self.sampled = [0] * self.num_train_points
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
                    if all(self.sampled):
                        epochs_done += 1
                        del self.sampled
                        self.sampled = [0] * self.num_train_points

        # Make sure that all the samples have been taken by the batch creator
        for sample_queue in self.ds.samples:
            while sample_queue.full():
                continue

        # Insert a number into sample_creator_done queue to mark completion
        self.ds.sample_creator_done.put(1)

        # Allow some time to copy over the sample
        time.sleep(2)

    def create_sample(self, sample_queue):
        points = []
        # Tracks the index of all points
        i = 0
        # Tracks the index of all points considered for sampling
        j = 0
        while j < self.sample_size and i < self.num_train_points:
            if not self.replace and self.sampled[i]:
                i += 1
                continue
            points.append(i)
            j += 1
            i += 1

        while i < self.num_train_points:
            if not self.replace and self.sampled[i]:
                i += 1
                continue
            p = random.randint(0, j)
            if p < self.sample_size:
                points[p] = i
            i += 1
            j += 1

        reservoir = self.ds.build_reservoir_sample(points)
        sample_queue.put(reservoir)
        # Loop over sampled after putting the reservoir in the queue
        for point in points:
            self.sampled[point] = 1
        del reservoir
