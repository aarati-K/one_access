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
        cur_sample = None
        while not self.stop_batch_creator.is_set():
            if self.batches.full():
                print("Waiting for client to take a batch from batches array")
                continue
            elif cur_sample is None and self.ds.samples.empty():
                print("Waiting for sample creator to create a sample")
                continue
            else:
                i = 0
                if cur_sample is None:
                    cur_sample = self.ds.samples.get()

                cur_batch = []
                while i < self.batch_size:
                    cur_batch.append(cur_sample[i+self.offset])
                    i += 1

                self.offset += i
                if self.offset == self.ds.sample_size:
                    cur_sample = None
                    self.offset = 0

                print(cur_batch)
                self.batches.put(np.array(cur_batch))
