from store.coco import CocoDetection
import time

batch_size = 1
ds = CocoDetection(input_data_folder="/home/aarati/datasets/coco/train2017",
                   annFile="/home/aarati/datasets/coco/annotations/instances_train2017.json",
                   batch_size=batch_size,
                   rel_sample_size=10000)
ds.count_num_points()
all_times = []
for i in range(1):
    start = time.time()
    ds.initialize_samples()
    end = time.time()
    all_times.append(end-start)
    s = ds.samples[0].get()
print(all_times)

# Sample creation time for sample size:
# 1: [1.12, 1.37, 2.32, 1.31, 2.42, 3.29, 2.92, 0.29, 1.53, 1.08]
# 10: [7.6, 11.75, 7.53, 12.64, 10.5, 13.93, 8.56, 10.16, 7.23, 13.1]
# 100: [24.1, 24.96, 24.9, 25.04, 25.07, 25.06, 25.25, 24.1, 22.52, 25.6]
# 1000: [26.35, 26.56, 26.44, 26.27, 26.31, 26.39, 27.25, 27.18, 26.43, 26.34]
# 10000: [28.3, 29.56, 29.55, 29.42, 29.09, 29.09, 29.36, 29.34, 29.8, 29.36]
