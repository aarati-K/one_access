from sampling.sample_creator import SampleCreator
from sampling.batch_creator import BatchCreator
from torch.multiprocessing import Event


class DataLoader:

    def __init__(self, data_store):
        """
          Start the batchCreator and sampleCreator.
          Read the memory config file, and create the right number
          of processes.
        """
        self.ds = data_store
        # Event to stop batch creator and sample creator
        self.stop_sc = Event()
        self.stop_bc = Event()

        # Start separate processes for sample_creator(s)
        self.sc = SampleCreator(self.ds, self.stop_sc)
        self.sc.start()

        # Start batch_creator(s)
        self.bc = BatchCreator(self.ds, self.stop_bc)
        self.bc.start()

    def get_next_batch(self):
        """
         Get the next batch from the queue
        """
        # Access batches from data_store.batches and return the batch
        # delete the used batch

        return self.ds.batches.get()

    def stop_batch_creation(self):
        self.stop_bc.set()
        self.stop_sc.set()
        # Wait for child processes to end
        self.bc.join()
        self.sc.join()
