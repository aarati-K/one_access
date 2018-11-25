import torch.multiprocessing as mp

class DataLoader():
  self.data_store = None
  self.batch_size = 0

  def __init__(self, data_store, batch_size):
    """
      Start the batchCreator and sampleCreator.
      Read the memory config file, and create the right number
      of processes.
    """
    # Mark data_store as shared memory
    # Start separate processes for batch_creator and sample_creator(s),
    # based on the memory config
    pass

  def next(self):
    """
     Get the next batch from the queue
    """
    # Access batches from data_store.batches and return the batch
    # delete the used batch
    pass
