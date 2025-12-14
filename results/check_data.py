import numpy as np

train_obj = np.load("train_scores.npy",allow_pickle=True).item()

print(train_obj.keys())


print(train_obj[471])
