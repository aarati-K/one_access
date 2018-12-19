from store.coco import CocoDetection
from load.load import DataLoader
import torchvision.transforms as transforms
import time
import numpy as np
import torch
from pathlib import Path


batch_size = 32
rel_sample_size = 400
ds = CocoDetection(
    input_data_folder="/home/aarati/datasets/coco/train2017",
    annFile="/home/aarati/datasets/coco/annotations/instances_train2017.json",
    rel_sample_size=rel_sample_size,
    batch_size=batch_size,
    max_batches=2,
    max_samples=1,
)

coco = ds.coco


# # HACK
def target_transform(img_id):
    if type(img_id) != int:
        img_id = img_id.tolist()
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    return anns


ds.target_transform = target_transform

start = time.time()
ds.initialize()
dl = DataLoader(ds)
end = time.time()
total_time = end-start
all_times = []

i = 0
while True:
    try:
        start = time.time()
        d, l = dl.get_next_batch()
        end = time.time()
        total_time += (end-start)
        all_times.append(end-start)
        i += 1
        if i%100 == 0:
            print(i)
    except:
        print("Finished epoch")
        break

print(total_time)
print(i)
dl.stop_batch_creation()

f = Path('/home/aarati/workspace/one_access/benchmarks/coco/measurements/one_access_{}_full_epoch.npy'.format(rel_sample_size)).open('wb')
np.save(f, np.array(all_times))
f.close()
# Total time: 385 ~ 6.4 min
