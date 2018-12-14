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
        if self.ds.batches.empty() and self.sc.exitcode:
            # Sample creator process has ended and batch creator is waiting on self.ds.samples
            print("Epochs completed, stopping batch creation")
            self.stop_batch_creation()
            return None

        # Access batches from data_store.batches and return the batch
        return self.ds.batches.get()

    def stop_batch_creation(self):
        # Attempt to gracefully terminate the processes
        self.stop_bc.set()
        self.stop_sc.set()
        time.sleep(2)

        # Terminate the processes forcefully
        self.bc.terminate()
        self.sc.terminate()

        # Wait for child processes to end
        self.bc.join()
        self.sc.join()
