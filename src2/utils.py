# from src2.parser import *
import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import datetime
from tqdm import tqdm
from pathlib import Path


class color:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def reconstruction_loss(x_hat, x, loss_type="smoothl1"):
    if loss_type == "mse":
        crit = nn.MSELoss(reduction="none")
    elif loss_type == "smoothl1":
        crit = nn.SmoothL1Loss(reduction="none")  # beta 기본값 1.0
    else:
        raise ValueError(loss_type)
    return crit(x_hat, x)


def convert_to_windows(data, model, cfg):
    windows = []
    w_size = model.n_window  # 10
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size : i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w if "TranAD" in cfg["model"]["name"] else w.view(-1))
    return torch.stack(windows)


def snippet_score(
    model, snippet, cfg, L=10, device="cuda", reduce="topk", k_ratio=0.05, p=95
):
    """
    하나의 스니펫(정규화된 (T,F))에 대한 anomaly score
    reduce:
      - "mean": 전체 평균
      - "topk": 상위 k% 평균
      - "percentile": 지정 분위수
    """
    model_device = next(model.parameters()).device
    if device is None:
        device = model_device
    else :
        device = torch.device(device)

    loss_type = cfg["training"]["loss_type"]

    s = snippet.to(device=device, dtype=torch.float32)
    window = convert_to_windows(s, model, cfg)
    src = window.permute(1, 0, 2)
    tgt = src[-1, :, :].unsqueeze(0)

    # print(src.shape,tgt.shape)
    out = model(src, tgt)

    if isinstance(out, tuple):
        x1, x2 = out
    else:
        x2 = out

    # 구현에 따라 x2가 shape가 (1,B,F)또는 (L,B,F)일 수 있음.
    """
    if x2.shape[0] != 1:
        x2_last = x2[-1, :, :]
    else:
        x2_last = x2
    """
    x2_last = x2
    # (1,B,F) -> (B,)윈도우별 score
    res = reconstruction_loss(x2_last, tgt, loss_type=loss_type)  # (1,B,F)
    score_w = res.mean(dim=2).view(-1)  # (B,)

    B = score_w.numel()

    if reduce == "mean":
        return score_w.mean().item()
    elif reduce == "topk":
        k = max(1, int(B * k_ratio))
        topk_vals, _ = torch.topk(score_w, k)
        return topk_vals.mean().item()
    elif reduce == "percentile":
        return torch.quantile(score_w, p / 100).item()
    else:  # max
        return score_w.max().item()


def fit_threshold(
    model, normal_snippets, cfg, L=10, device="cuda", reduce="topk", q=90, p=95
):
    scores = [
        snippet_score(model, s, cfg=cfg, L=L, device=device, reduce=reduce, p=p)
        for s in tqdm(normal_snippets)
    ]
    thr = float(np.percentile(scores, q))
    return thr, np.array(scores)


def convert_to_windows_mod(data, cfg, model="TranAD"):

    windows = []
    # print(data.shape)
    for X in data:
        window = []
        w_size = model.n_window  # 10
        for i, g in enumerate(X):
            if i >= w_size:
                w = X[i - w_size : i]
            else:
                w = torch.cat([X[0].repeat(w_size - i, 1), X[0:i]])
            window.append(w if "TranAD" in cfg["model"]["name"] else w.view(-1))
        window_tensor = torch.stack(window, dim=0)
        windows.append(window_tensor)
    return torch.stack(windows, dim=0)


def load_model(modelname,device, dims, args, cfg):
    import src2.models
    
    model_class = getattr(src2.models, modelname)
    model = model_class(dims, cfg).float()
    model = model.to(device).float()
    # optimizer
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    opt_type = opt_cfg.get("type", "adamw").lower()

    lr = opt_cfg.get("lr", getattr(model, "lr", 1e-3))
    weight_decay = opt_cfg.get("weight_decay", 1e-5)

    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    # Scheduler
    sch_cfg = cfg.get("training", {}).get("scheduler", {})
    sch_type = sch_cfg.get("type", "steplr").lower()

    if sch_type == "steplr":
        step_size = sch_cfg.get("step_size", 5)
        gamma = sch_cfg.get("gamma", 0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif sch_type == "none":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)
    else:
        raise ValueError(f"Unsupported scheduler type: {sch_type}")

    config_path = Path(args.config)
    run_id = config_path.stem
    folder = Path("./checkpoints") / run_id
    fname = f"{folder}/model.ckpt"

    if os.path.exists(fname):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        accuracy_list = checkpoint["accuracy_list"]
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def save_model(model, optimizer, scheduler, epoch, accuracy_list, args):
    config_path = Path(args.config)
    run_id = config_path.stem

    folder = Path("./checkpoints") / run_id
    folder.mkdir(parents=True, exist_ok=True)

    file_path = f"{folder}/model.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "accuracy_list": accuracy_list,
        },
        file_path,
    )


def load_dataset(dataset, test=False):
    folder = os.path.join(dataset)
    if not os.path.exists(folder):
        raise Exception("Processed Data not found.")
    loader = []
    for file in ["train", "test", "labels", "train_labels"]:
        loader.append(np.load(os.path.join(folder, f"{file}.npy"), allow_pickle=True))
    # loader = [i[:, debug:debug+1] for i in loader]

    train_loader = DataLoader(loader[0][:, :, :], batch_size=loader[0].shape[0])

    # test가 True이면, test data를 받아온다.
    test_loader = loader[1].item()
    labels = loader[2].item()

    train_data_dict = np.load(
        os.path.join(folder, "train_dict.npy"), allow_pickle=True
    ).item()

    train_labels = loader[3].item()
    # 전체 데이터를 데이터로터 타입으로 변환
    return train_loader, test_loader, labels, train_data_dict, train_labels
