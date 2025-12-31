import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from collections import Counter
import random
from processdata.utils import *


brand = [0, 1, 2, 3]
brand_num = 1  # change this number

# brand1/2
test_obj = glob(f"./data/battery_brand{brand[brand_num]}/test/*.pkl")
train_obj = glob(f"./data/battery_brand{brand[brand_num]}/train/*.pkl")
loaded_obj = train_obj + test_obj

# brand3
# loaded_obj = glob("./data/battery_brand3/data/*.pkl")

# print(loaded_obj)
car_data, labels = sort_by_car(loaded_obj)
# Add dx features
car_data = add_dx_features(car_data)

for k, v in labels.items():
    if v == 10:
        labels[k] = 1
print(len(labels.keys()))

for key in car_data:
    car_data[key] = np.stack(car_data[key], axis=0)
print("예시 car shape : ", next(iter(car_data.values())).shape)

# normal / abnormal 먼저 구분
normal_ids = [cid for cid, lab in labels.items() if lab == 0]
abnormal_ids = set([cid for cid, lab in labels.items() if lab == 1])
print("normal car 수 :", len(normal_ids))
print("abnormal car 수 :", len(abnormal_ids))


# random seed (0)
random.seed(0)

# train / test split
# normal : 10 % test, 나머지 train
# abnormal : all test


# normal split
k_test_normal = int(len(normal_ids) * 0.2)
test_normal_ids = set(random.sample(normal_ids, k_test_normal))

train_normal_ids = set([cid for cid in normal_ids if cid not in test_normal_ids])
# random.shuffle(remaining_normal)

# k_val_normal = int(len(remaining_normal) * 0.2)
# val_normal_ids = set(remaining_normal[:k_val_normal])
# train_normal_ids = set(remaining_normal[k_val_normal:])

# abnormal split
# random.shuffle(abnormal_ids)
# k_val_abnormal = max(1, int(len(abnormal_ids)*0.3))
# val_abnormal_ids = set(abnormal_ids[:k_val_abnormal])
# test_abnormal_ids = set(abnormal_ids[k_val_abnormal:])

# id collection
train_ids = train_normal_ids
# val_ids = val_normal_ids | val_abnormal_ids
test_ids = test_normal_ids | abnormal_ids

assert len(train_ids & test_ids) == 0

print("=== Split 결과 (car 기준) ===")
print("Train normal:", len(train_ids), "abnormal: 0")
# print("Val   normal:", len(val_normal_ids), "abnormal:", len(val_abnormal_ids))
print("Test  normal:", len(test_normal_ids), "abnormal:", len(abnormal_ids))


train_data = {cid: car_data[cid] for cid in train_ids}
train_labels = {cid: labels[cid] for cid in train_ids}


test_data = {cid: car_data[cid] for cid in test_ids}
test_labels = {cid: labels[cid] for cid in test_ids}

print("Train label 분포:", Counter(train_labels.values()))
print("Test  label 분포:", Counter(test_labels.values()))


# 1. train_data 기준으로 normalize feature 계산 (Dyad 기반)
mean_vals, std_vals = compute_mean_std(train_data)
min_vals, max_vals = compute_min_max(train_data)
train_data_norm = normalize_dict(train_data, mean_vals, std_vals, max_vals, min_vals)
test_data_norm = normalize_dict(test_data, mean_vals, std_vals, max_vals, min_vals)
# train_data_norm = minmax_normalize_dict(train_data, min_vals, max_vals)
# test_data_norm = minmax_normalize_dict(test_data, min_vals, max_vals)

# 2. train/test normalize (quantile min-max 방식)
# low_vals, high_vals = compute_min_max_quantile(train_data)
# train_data_norm = lowhigh_norm_dict_quantile(train_data,low_vals,high_vals)
# test_data_norm = lowhigh_norm_dict_quantile(test_data,low_vals,high_vals)

# 3. z-normalize
# mean_vals, std_vals = compute_mean_std(train_data)
# train_data_norm = meanstd_normalize_dict(train_data,mean_vals,std_vals)
# test_data_norm = meanstd_normalize_dict(test_data,mean_vals,std_vals)

# current_idx = 1
train_snippets = []
for cid, arr in train_data_norm.items():
    for i in range(arr.shape[0]):
        # current z-norm 진행
        # cur = arr[i][:,current_idx]
        # mean = cur.mean()
        # std = cur.std() + 1e-8
        # arr[i][:,current_idx] = (cur-mean)/std
        train_snippets.append(arr[i])

train_snippets = np.array(train_snippets)
print("Train_snippets shape : ", train_snippets.shape)

# Current Z-norm
"""
test_snippets = {}
for cid , arr in test_data_norm.items():
    for i in range(arr.shape[0]):
        cur = arr[i][:,current_idx]
        mean = cur.mean()
        std = cur.std() + 1e-8
        arr[i][:,current_idx] = (cur-mean)/std
    test_snippets[cid] = arr
"""

np.save(
    f"train_brand{brand[brand_num]}.npy",
    train_snippets,
    allow_pickle=True,
)

np.save(
    f"train_labels_brand{brand[brand_num]}.npy",
    train_labels,
    allow_pickle=True,
)

np.save(
    f"test_brand{brand[brand_num]}.npy",
    test_data_norm,
    allow_pickle=True,
)
np.save(
    f"test_labels_brand{brand[brand_num]}.npy",
    test_labels,
    allow_pickle=True,
)

np.save(
    f"minmax_stats_brand{brand[brand_num]}.npy",
    {"min": min_vals, "max": max_vals},
    allow_pickle=True,
)

np.save(
    f"train_brand{brand[brand_num]}_dict.npy",
    train_data_norm,
    allow_pickle=True,
)
