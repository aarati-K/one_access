class BatchCreator():

  def __init__(self, data_store, batch_size):
    self.data_store = data_store
    self.batch_size = batch_size

  def create_batches(self):
    """
      Keep creating batches in the background
    """
    # Read the samples on memory (data_store.samples) and generate batches.
    # Place the batches in data_store.batches.
    # Pause if data_store.batches is full (or if data_store.samples empty)
    pass