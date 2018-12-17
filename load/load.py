from sampling.sample_creator import SampleCreator
from sampling.batch_creator import BatchCreator
from torch.multiprocessing import Event
import time


class DataLoader:

    def __init__(self, data_store, epochs=1):
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
        self.epochs = epochs
        self.sc = SampleCreator(self.ds, event=self.stop_sc, epochs=self.epochs, sampled=self.ds.points_sampled)
        self.sc.start()

        # Start batch_creator(s)
        self.bc = BatchCreator(self.ds, self.stop_bc)
        self.bc.start()

    def get_next_batch(self):
        """
         Get the next batch from the queue
        """
        if self.ds.batch_creator_done.full():
            return None

        # Access batches from data_store.batches and return the batch
        try:
            batch = self.ds.batches.get()
            return batch
        except Exception as e:
            print(e)
            return None

    def stop_batch_creation(self):
        # Attempt to gracefully terminate the processes
        self.stop_bc.set()
        self.stop_sc.set()
        time.sleep(1)

        # Terminate the processes forcefully
        self.bc.terminate()
        self.sc.terminate()

        # Wait for child processes to end
        self.bc.join()
        self.sc.join()
