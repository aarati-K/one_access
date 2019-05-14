from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
import os
from torchvision import transforms
import numpy as np

from store.memory_hierarchy import StorageAttributes, StorageComponents
from store.store import DataStore, Metadata, MetadataField


class CocoDetection(DataStore):
    def __init__(self, annFile, **kwargs):
        super(CocoDetection, self).__init__(**kwargs)
        self.dataset_name = "ms-coco-train-2017"
        self.metadata = Metadata(self).load()
        self.created = False
        train_metadata = self.metadata.get(self.TRAIN_FOLDER)
        if train_metadata:
            self.key_size = train_metadata.get(MetadataField.KEY_SIZE)
            self.value_size = train_metadata.get(MetadataField.VALUE_SIZE)
            self.created = True
        else:
            self.key_size = self.value_size = None

        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.ids.sort()
        # Tentative number of files, might change
        self.num_train_files = 10
        self.num_points_in_numpy_batch = 500

    def count_num_points(self):
        self.num_train_points = len(self.ids)

    def generate_IR(self):
        if self.created:
            print("IR already created in folder ", self.get_data_folder_path())
            return

        data_folder_path = self.get_data_folder_path()
        if not Path(data_folder_path).exists():
            print("Creating directory(s)", data_folder_path)
            Path(data_folder_path).mkdir(parents=True, exist_ok=True)

        # Create train and test directories
        train_folder_path = data_folder_path + '/' + self.TRAIN_FOLDER
        test_folder_path = data_folder_path + '/' + self.TEST_FOLDER
        for path in [train_folder_path, test_folder_path]:
            if not Path(path).exists():
                print("Creating directory", path)
                Path(path).mkdir(parents=True, exist_ok=True)

        num_points_in_numpy_batch = self.num_points_in_numpy_batch
        numpy_batches = []
        cur_numpy_batch = []
        i = 0
        while i < self.num_train_points:
            cur_numpy_batch.append(i)
            i += 1
            if i % num_points_in_numpy_batch == 0:
                numpy_batches.append(cur_numpy_batch)
                cur_numpy_batch = []

        # Incomplete batch
        if len(cur_numpy_batch) != 0:
            numpy_batches.append(cur_numpy_batch)

        num_batches_in_file = int(len(numpy_batches)/self.num_train_files + 1)
        batches_in_files = []
        cur_file_batches = []
        i = 0
        while i < len(numpy_batches):
            cur_file_batches.append(numpy_batches[i])
            i += 1
            if i % num_batches_in_file == 0:
                batches_in_files.append(cur_file_batches)
                cur_file_batches = []

        if i % num_batches_in_file != 0:
            batches_in_files.append(cur_file_batches)

        # Distributed the batches roughly over files, might require
        # lesser number of files
        if len(batches_in_files) != self.num_train_files:
            self.num_train_files = len(batches_in_files)

        # Write metadata before creating files
        self.write_metadata(batches_in_files)

        for i in range(len(batches_in_files)):
            fname = train_folder_path + '/' + self.DATA_FILE.format(i)
            f = Path(fname).open('ab')
            for batch in batches_in_files[i]:
                # create a batch
                nparr = [[self.get_image(i), self.ids[i]] for i in batch]
                nparr = np.array(nparr)
                np.save(f, nparr)
            f.close()
            print("Finished creating file ", fname)

    def get_image(self, index):
        """
        :param index: Index of the image to get
        :return: a PIL RGB image object of shape 224*224*3
        """
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.input_data_folder, path)).convert('RGB')
        # NOTE:Crop the image. Hardcoded for now.
        img = transforms.RandomResizedCrop(224)(img)
        img = np.array(img)
        img = img.reshape(3, 224, 224)
        return img

    def write_metadata(self, batches_in_files):
        metadata_dict = {}
        train_metadata = {
            MetadataField.KEY_SIZE: 4, # bytes
            MetadataField.VALUE_SIZE: 0, # Does not matter
        }
        train_files = {}
        for i in range(self.num_train_files):
            batches = batches_in_files[i]
            chunk_count = len(batches)
            kv_count = 0
            for batch in batches:
                kv_count += len(batch)
            train_files[self.DATA_FILE.format(i)] = {
                MetadataField.KV_COUNT: kv_count,
                MetadataField.CHUNK_COUNT: chunk_count
            }
        train_metadata[MetadataField.FILES] = train_files
        metadata_dict[self.TRAIN_FOLDER] = train_metadata

        metadata = Metadata(self)
        metadata.store(metadata_dict)
        self.metadata = metadata.load()

        # Skipping test metadata, not generating IR for test files

    def get_data_folder_path(self):
        return self.mem_config.get(StorageComponents.HDD)\
            .get(StorageAttributes.ONE_ACCESS_DIR) + '/' + self.dataset_name
