import torch.multiprocessing as mp


class DataLoader():

  def __init__(self, data_store, batch_size=10):
    """
      Start the batchCreator and sampleCreator.
      Read the memory config file, and create the right number
      of processes.
    """
    self.data_store = data_store
    self.batch_size = batch_size
    # Mark data_store as shared memory
    # Start separate processes for batch_creator and sample_creator(s),
    # based on the memory config

  def get_next_batch(self):
    """
     Get the next batch from the queue
    """
    # Access batches from data_store.batches and return the batch
    # delete the used batch
    pass
