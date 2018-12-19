from PIL import Image
from time import sleep
from torch.multiprocessing import Process
import numpy as np
import torch
import random
import torchvision.transforms as transforms
import time


class BatchCreator(Process):
    """
    Responsible for creating batches from the created samples
    """

    def __init__(self, data_store, event):
        super(BatchCreator, self).__init__()
        self.ds = data_store
        self.batch_size = self.ds.batch_size
        self.is_stop = False
        self.max_batches = self.ds.max_batches
        self.stop_batch_creator = event
        self.target_transform = self.ds.target_transform
        self.transform = self.ds.transform

    def run(self):
        """
        Keep creating the batches in the background
        """
        cur_sample = None
        cur_order = None
        offset = 0
        while not self.stop_batch_creator.is_set():
            if self.ds.batches.full():
                continue
            for sample_queue in self.ds.samples:
                if cur_sample is None and sample_queue.empty():
                    continue
                else:
                    i = 0
                    if cur_sample is None:
                        cur_sample = sample_queue.get()
                        cur_order = list(range(len(cur_sample[0])))
                        random.shuffle(cur_order)

                    cur_batch_data = []
                    cur_batch_labels = []
                    while i < self.batch_size and offset < len(cur_sample[0]):
                        index = cur_order[offset]
                        cur_batch_data.append(cur_sample[0][index])
                        cur_batch_labels.append(cur_sample[1][index])
                        i += 1
                        offset += 1

                    cur_batch_data = np.array(cur_batch_data)
                    cur_batch_data = torch.from_numpy(cur_batch_data)

                    if self.transform:
                        cur_batch_data_ = []
                        for img_tensor in cur_batch_data:
                            img = transforms.ToPILImage()(img_tensor)
                            img = self.transform(img)
                            cur_batch_data_.append(img)
                        cur_batch_data = torch.stack(cur_batch_data_)

                    cur_batch_labels = np.array(cur_batch_labels)
                    cur_batch_labels = torch.from_numpy(cur_batch_labels) 
                    if self.target_transform:
                        cur_batch_labels_ = []
                        for img_tensor in cur_batch_labels:
                            img = self.target_transform(img_tensor)
                            cur_batch_labels_.append(img)
                        try:
                            cur_batch_labels = torch.stack(cur_batch_labels_)
                        except:
                            # HACK
                            cur_batch_labels = cur_batch_labels_

                    self.ds.batches.put((cur_batch_data, cur_batch_labels))

                    if offset == len(cur_sample[0]):
                        cur_sample = None
                        offset = 0
                        # Check if sample creator has finished running
                        if self.ds.sample_creator_done.full():
                            # Wait on batches to become empty
                            while not self.ds.batches.empty():
                                continue

                            # Mark the batch creator as done
                            self.ds.batch_creator_done.put(1)

                            # Allow some time to copy over the batch
                            time.sleep(2)

                            return
