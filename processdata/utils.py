from tqdm import tqdm
import numpy as np
import torch


def sort_by_car(loaded_obj):
    labels = {}
    car_data = {}
    for path in tqdm(loaded_obj):
        obj = torch.load(path)
        key = int(obj[1]["car"])
        label = int(obj[1]["label"])
        if key not in car_data.keys():
            labels[key] = label
            car_data[key] = [obj[0]]
        else:
            car_data[key].append(obj[0])

    return car_data, labels

def add_dx_features(car_data):
    """
    ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp']
    """
    new_car_data = {}

    for cid, val in car_data.items():
        new_car_data[cid]=[]
        for data in val:
            x = data[:,[1]] # current col

            dx = np.empty_like(x)
            dx[:-1] = x[1:] - x[:-1]
            dx[-1] = x[-1] - x[-2]
            data_aug = np.concatenate([data,dx],axis=1)
            new_car_data[cid].append(data_aug)
    return new_car_data



def compute_mean_std(train_data):
    all_data = np.concatenate(list(train_data.values()), axis=0)
    mean_vals = all_data.mean(axis=(0, 1))
    std_vals = all_data.std(axis=(0, 1)) + 1e-8
    return mean_vals, std_vals


def meanstd_normalize_dict(data_dict, mean_vals, std_vals):
    norm_dict = {}

    for cid, data in data_dict.items():
        norm = (data - mean_vals) / std_vals
        norm_dict[cid] = norm
    return norm_dict


def normalize_dict(data_dict, mean_vals, std_vals, max_vals, min_vals):
    norm_dict = {}

    for cid, data in data_dict.items():
        norm = (data - mean_vals) / np.maximum(
            np.maximum(1e-4, std_vals), 0.1 * (max_vals - min_vals)
        )
        norm_dict[cid] = norm
    return norm_dict


def compute_min_max(train_data):
    """
    train_data: {car_id: np.array(shape=(N_i, 128, F)), ...}
    return: (min_vals, max_vals)  # shape (F,), (F,)
    """
    min_vals = None
    max_vals = None

    for data in train_data.values():
        # data: (N_i, 128, F)
        # 전체 시점/스니펫을 한꺼번에 보고 feature별 min/max
        # axis=(0,1) -> N_i, 128 차원에 대해 축소, F만 남김
        cur_min = data.min(axis=(0, 1))
        cur_max = data.max(axis=(0, 1))

        if min_vals is None:
            min_vals = cur_min
            max_vals = cur_max
        else:
            min_vals = np.minimum(min_vals, cur_min)
            max_vals = np.maximum(max_vals, cur_max)

    return min_vals, max_vals


def compute_min_max_quantile(train_data):
    """
    train_data: {car_id: np.array(shape=(N_i, 128, F)), ...}
    return: (low_vals, high_vals)  # shape (F,), (F,)
    """

    cur_min = []
    cur_max = []
    for data in train_data.values():
        cur_min.append(data.min(axis=(0, 1)))
        cur_max.append(data.max(axis=(0, 1)))

    low_vals = np.quantile(cur_min, 0.01, axis=0)  # (F,)
    high_vals = np.quantile(cur_max, 0.99, axis=0)  # (F,)
    return low_vals, high_vals


def lowhigh_norm_dict_quantile(data_dict, low_vals, high_vals, eps=1e-8):
    """
    data_dict: {car_id: np.array(...)}
    low_vals, high_vals: shape (F,)
    """
    clipped_dict = {}
    for car_id, data in data_dict.items():
        data_clipped = np.clip(data, low_vals, high_vals)
        clipped_dict[car_id] = data_clipped

    norm_dict = {}
    denom = (high_vals - low_vals).copy()
    denom[denom == 0] = 1.0

    for car_id, data in clipped_dict.items():
        norm = (data - low_vals) / (denom + eps)
        norm_dict[car_id] = norm
    return norm_dict


def minmax_normalize_dict(data_dict, min_vals, max_vals, eps=1e-8):
    """
    data_dict: {car_id: np.array(...)}
    min_vals, max_vals: shape (F,)
    """
    norm_dict = {}

    # 분모 0 방지
    denom = (max_vals - min_vals).copy()
    denom[denom == 0] = 1.0  # 변화가 없는 feature는 그대로 0이 되도록

    for car_id, data in data_dict.items():
        # data: (N, 128, F)

        norm = (data - min_vals) / (denom + eps)
        norm_dict[car_id] = norm

    return norm_dict
