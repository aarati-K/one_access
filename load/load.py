import torch.multiprocessing as mp
from sampling.sample_creator import SampleCreator


class DataLoader():

  def __init__(self, data_store, batch_size=10):
    """
      Start the batchCreator and sampleCreator.
      Read the memory config file, and create the right number
      of processes.
    """
    self.data_store = data_store
    self.batch_size = batch_size
    
    # Start separate processes for sample_creator(s)
    sc = SampleCreator(self.data_store)
    sc.start()
    
    # Start batch_creator(s)

    # Wait for sub-processes to terminate
    sc.join()

  def get_next_batch(self):
    """
     Get the next batch from the queue
    """
    # Access batches from data_store.batches and return the batch
    # delete the used batch
    pass
    
