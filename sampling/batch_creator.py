from time import sleep

from torch.multiprocessing import Process
import numpy as np


class BatchCreator(Process):
    """
    Responsible for creating batches from the created samples
    """

    def __init__(self, data_store, event):
        super(BatchCreator, self).__init__()
        self.ds = data_store
        self.batch_size = self.ds.batch_size
        self.batches = self.ds.batches
        self.max_batches = self.ds.max_batches
        self.offset = 0
        self.stop_batch_creator = event
        self.is_stop = False

    def run(self):
        """
        Keep creating the batches in the background
        """
        while not self.stop_batch_creator.is_set():
            if self.ds.samples.empty():
                print("Waiting for sample creator to create a sample")
                continue
            elif self.batches.full():
                print("Waiting for client to take a batch from batches array")
                continue
            else:
                i = 0
                if not curr_sample or self.offset == self.ds.sample_size:
                    self.offset = 0
                    curr_sample = self.ds.samples.get()
                curr_batch = []
                while i < self.batch_size:
                    curr_batch.append(curr_sample[i+self.offset])
                    i = i + 1
                self.offset += i
                self.batches.put(np.array(curr_batch))
