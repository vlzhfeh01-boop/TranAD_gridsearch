import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from time import time
from pprint import pprint
from src2.utils import *
from src2.parser import *
from src2.models import *
import numpy as np
import json
import random

# Added
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def backprop(epoch, model, data, dataO, optimizer, scheduler, cfg, training=True):
    feats = dataO.shape[1]
    # Added
    # TrinD Shape = (N,128,10,8)
    if "TranAD" in model.name:
        w_size = model.n_window
        # mse = nn.MSELoss(reduction="none")
        n = epoch + 1
        if training:
            model.train()
            total_loss = 0.0
            count = 0
            data_x = torch.as_tensor(data, dtype=torch.double)

            batch_size = cfg["training"]["batch_size"]
            loss_type = cfg["training"]["loss_type"]
            dataloader = DataLoader(data_x, shuffle=True, batch_size=batch_size)

            for batch in tqdm(dataloader):
                optimizer.zero_grad()

                batch = convert_to_windows_mod(batch, cfg, model)

                B, N_win, L, F = (
                    batch.shape
                )  # Batch size, Number of window, snippet length, Feature
                batch_loss = 0.0  # 배치 평균 누적용

                for b in range(B):
                    snippet = batch[b]  # (128,10,8)
                    src = snippet.permute(1, 0, 2)  # (10,128,8)

                    tgt = src[-1, :, :].unsqueeze(0)
                    # forward per one snippet

                    out = model(src, tgt)  # return (x1,x2) or tensor

                    # loss 설정
                    if isinstance(out, tuple):
                        x1, x2 = out

                        assert x1.shape == tgt.shape and x2.shape == tgt.shape

                        # loss1 = mse(x1, tgt).mean()
                        # loss2 = mse(x2, tgt).mean()
                        loss1 = reconstruction_loss(x1, tgt, loss_type=loss_type).mean()
                        loss2 = reconstruction_loss(x2, tgt, loss_type=loss_type).mean()
                        loss = (1 / n) * loss1 + (1 - 1 / n) * loss2
                    else:
                        x_pred = out
                        loss = reconstruction_loss(
                            x_pred, tgt, loss_type=loss_type
                        ).mean()

                    # backward
                    (loss / B).backward()
                    batch_loss += loss.item()
                optimizer.step()

                total_loss += batch_loss / B
                count += 1

            scheduler.step()
            avg_loss = total_loss / max(1, count)
            tqdm.write(f"Epoch {epoch},\tL1 = {total_loss / max(1, count)}")
            return avg_loss, optimizer.param_groups[0]["lr"]
        else:

            q = cfg["scoring"]["percentile_q"]
            reduce_mode = cfg["scoring"]["reduce"]
            k_ratio = cfg["scoring"]["k_ratio"]
            car_p = ["scoring"]["car_positive_ratio"]
            threshold, train_scores = fit_threshold(
                model, trainO, reduce=reduce_mode, q=q
            )

            # Changed mainly
            car_scores = {}
            for cid, arr in data.items():
                scores = []
                arr = torch.as_tensor(arr[:, :, :])
                for i in range(arr.shape[0]):
                    score = snippet_score(
                        model, arr[i], reduce=reduce_mode, k_ratio=k_ratio, p=car_p
                    )
                    scores.append(score)
                car_scores[cid] = scores
            preds = {}
            for cid, scores in car_scores.items():
                # snippet-level -> car-level
                ratio = sum(s > threshold for s in scores) / len(scores)
                pred_car = 1 if ratio >= car_p else 0
                preds[cid] = pred_car
            return car_scores, preds, threshold, train_scores
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f"Epoch {epoch},\tMSE = {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]["lr"]
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == "__main__":
    args = get_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    if args.test:
        cfg["experiment"]["name"] = "Test"
    else:
        cfg["experiment"]["name"] = "Train"
    cfg["data"]["dataset"] = args.dataset
    cfg["model"]["name"] = args.model

    train_loader, test_loader, labels, train_dict = load_dataset(
        cfg["data"]["output_folder"] + args.dataset
    )
    # Batch Size = Entire Time Series Data (L) 전체 데이터를 받아온다.

    # Added
    trainD = next(iter(train_loader))
    trainO = trainD
    testD = test_loader
    testO = testD

    print(trainO.shape)
    print(type(testO))
    print(type(labels))

    model, optimizer, scheduler, epoch, accuracy_list = load_model(
        cfg["model"]["name"],
        trainO.shape[-1],
        cfg=cfg,  # modified # labels.shape[1]  # labels.shape[1] = dimensions
    )  # epoch = -1 , model 없는 경우

    ### Training phase
    if not args.test:
        print(f"{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}")
        num_epochs = cfg["training"]["num_epochs"]

        s = trainD
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):

            lossT, lr = backprop(
                e, model, s, trainO, optimizer, scheduler, cfg, training=True
            )

            accuracy_list.append((lossT, lr))
            print(f"Epoch {e} : ", accuracy_list[e])
        print(
            color.BOLD
            + "Training time: "
            + "{:10.4f}".format(time() - start)
            + " s"
            + color.ENDC
        )

        save_model(model, optimizer, scheduler, e, accuracy_list, cfg=cfg)
        # plot_accuracies(accuracy_list, f"{args.model}_{args.dataset}")

    ### Testing phase
    optimizer.zero_grad()
    model.eval()  # eval이기때문에 grad계산 X
    with torch.no_grad():

        print(
            f"{color.HEADER}Testing {cfg['model']['name']} on {cfg['data']['dataset']}{color.ENDC}"
        )
        # loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        # Added
        scores, preds, thr, train_scores = backprop(
            0, model, testD, trainO, optimizer, scheduler, training=False, cfg=cfg
        )
        print("Threshold : ", thr)
        train_scores, _, _, _ = backprop(
            0, model, train_dict, trainO, optimizer, scheduler, cfg, training=False
        )

    print("Save Score Files")
    np.save("./results/test_scores.npy", scores, allow_pickle=True)
    np.save("./results/train_scores.npy", train_scores, allow_pickle=True)

    y_true = np.array([labels[cid] for cid in labels])
    y_pred = np.array([preds[cid] for cid in labels])
    print("Evaluation Results for cars")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
