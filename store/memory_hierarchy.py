import json
import os


class StorageComponents():
    SSD = 'ssd'
    HDD = 'hdd'


class StorageAttributes():
    MOUNT_POINT = "mount_point"
    ONE_ACCESS_DIR = "one_access_dir"


class MemoryHierarchy():
    """
    Read the memory hierarchy specified in memory.config. By
    default, we assume that as much main_memory can be used as
    necessary/available. 
  """
    mem_config = {}

    @classmethod
    def load(cls):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(cur_dir, os.pardir, "memory.config")
        data = open(config_file).read()
        cls.mem_config = json.loads(data)

    @classmethod
    def get_config(cls, component):
        if component in cls.mem_config.keys():
            return cls.mem_config[component]
        return None
