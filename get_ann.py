import os
from glob import glob
path = '/home/node1/Desktop/code/ai/data/lrs2_preprocess/mvlrs_v1/pretrain_preprocessed/'
files = glob(os.path.join(path, '*/*'))
import numpy as np
files = np.array(files)
shuffled_indices = np.random.permutation(len(files))
train_size = int(len(files) * 0.6)
test_size = int(len(files) * 0.2)
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:train_size+test_size]
val_indices = shuffled_indices[train_size+test_size:]

train_data = files[train_indices]
test_data = files[test_indices]
val_data = files[val_indices]

prefix_len = len(path)

with open('filelists/train.txt', "w") as f:
    for data in train_data:
        f.write(f"{data[prefix_len:]}\n")

with open('filelists/test.txt', "w") as f:
    for data in test_data:
        f.write(f"{data[prefix_len:]}\n")

with open('filelists/val.txt', "w") as f:
    for data in val_data:
        f.write(f"{data[prefix_len:]}\n")