from sampling.sample_creator import SampleCreator
from sampling.batch_creator import BatchCreator
from multiprocessing import Event

class DataLoader():

    def __init__(self, data_store, batch_size=10):
        """
          Start the batchCreator and sampleCreator.
          Read the memory config file, and create the right number
          of processes.
        """
        self.data_store = data_store
        self.data_store.batch_size = batch_size
        # Event to stop batch creator and sample creator
        self.event = Event()

        # Start separate processes for sample_creator(s)
        sc = SampleCreator(self.data_store, self.event)
        sc.start()

        # Start batch_creator(s)
        bc = BatchCreator(self.data_store, self.event)
        bc.start()
        # Wait for sub-processes to terminate
        sc.join()

    def get_next_batch(self):
        """
         Get the next batch from the queue
        """
        # Access batches from data_store.batches and return the batch
        # delete the used batch
        pass
