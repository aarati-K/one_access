from time import sleep

import torch
from torch.multiprocessing import Process
import numpy as np
from torchvision import transforms


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
                continue
            elif cur_sample is None and self.ds.samples.empty():
                continue
            else:
                i = 0
                if cur_sample is None:
                    cur_sample = self.ds.samples.get()

                cur_batch_data = []
                cur_batch_labels = []
                while i < self.batch_size:
                    cur_batch_data.append(cur_sample[0][i+self.offset])
                    cur_batch_labels.append(cur_sample[1][i+self.offset])
                    i += 1

                self.offset += i
                if self.offset == self.ds.sample_size:
                    cur_sample = None
                    self.offset = 0

                # Apply transforms
                cur_batch_data = np.array(cur_batch_data)
                cur_batch_data = transforms.ToTensor()(cur_batch_data)
                cur_batch_data = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(cur_batch_data)

                cur_batch_labels = np.array(cur_batch_labels)
                cur_batch_labels = torch.from_numpy(cur_batch_labels)

                self.batches.put((cur_batch_data, cur_batch_labels))
