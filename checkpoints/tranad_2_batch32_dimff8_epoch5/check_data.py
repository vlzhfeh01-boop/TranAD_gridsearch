import torch

file = torch.load("model.ckpt")

print(type(file))
print(file.keys())

print(file["accuracy_list"])


for i,g in enumerate(file["accuracy_list"]):
    print(f"epoch : {i+1}, loss : {g[0]}, LR : {g[1]}")
