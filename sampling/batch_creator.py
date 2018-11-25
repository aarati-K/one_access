class BatchCreator():
  self.data_store = None
  self.batch_size = 0

  def __init__(self, data_store, batch_size):
    pass

  def create_batches(self):
    """
      Keep creating batches in the background
    """
    # Read the samples on memory (data_store.samples) and
    # generate batches.
    # Place the batches in data_store.batches.
    # Pause if data_store.batches is full.
    pass