from store.coco import CocoDetection
import time

start = time.time()
# Do not create samples, only generate IR
ds = CocoDetection(input_data_folder="/home/aarati/datasets/coco/train2017",
                   annFile="/home/aarati/datasets/coco/annotations/instances_train2017.json")
ds.count_num_points()
ds.generate_IR()
end = time.time()
print(end-start)
