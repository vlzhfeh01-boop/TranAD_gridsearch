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
from scoring.score import *
from pathlib import Path

# Added
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def backprop(
    epoch,
    model,
    data,
    testD,
    train_dict_D,
    dataO,
    optimizer,
    scheduler,
    cfg,
    training=True,
):
    feats = dataO.shape[1]
    # Added
    # TranAD Shape = (N,128,10,8)
    if "TranAD" in model.name:
        w_size = cfg["model"]["n_window"]
        # mse = nn.MSELoss(reduction="none")
        n = epoch + 1
        if training:
            model.train()
            total_loss = 0.0
            count = 0
            data_x = torch.as_tensor(data, dtype=torch.float32)

            #data_x = convert_to_windows_mod(data_x,cfg)

            #print("Nan : ", torch.isnan(data_x).any())
            #print("Inf : ",torch.isinf(data_x).any())
            #print(" std : ",data_x.std())
            #print("std per feature : ", data_x.std(dim=2))
            #print("min/max per feature : ", data_x.min(dim=2),data_x.max(dim=2))

            batch_size = cfg["training"]["batch_size"]
            loss_type = cfg["training"]["loss_type"]
            dataloader = DataLoader(data_x, shuffle=True, batch_size=batch_size,num_workers=4,pin_memory=True,persistent_workers=True)
            for batch in tqdm(dataloader):

                optimizer.zero_grad()
                batch = batch.to(device, non_blocking=True)

                batch = convert_to_windows_mod(batch, cfg, model)
                #batch = convert_to_windows_unfold(batch, cfg)

                batch = batch[:,:,:,1:] #exclude volt

                B, T, L, F = (
                    batch.shape
                )  # Batch size, Number of window, window length, Feature

                if count == 0:
                    print("batch.shape:", batch.shape, " w_size:", w_size)
                    count += 1

                src = (
                    batch.permute(2, 0, 1, 3).reshape(L,B*T,F)
                )  # (10,128,8)

                tgt = src[-1, :, :].unsqueeze(0)
                # forward per one snippet
                out = model(src, tgt)  # return (x1,x2) or tensor

                #print("batch device:", batch.device)
                #print("model device:", next(model.parameters()).device)

                # loss 설정
                if isinstance(out, tuple):
                    x1, x2 = out

                    #assert x1.shape == tgt.shape and x2.shape == tgt.shape
                    # loss1 = mse(x1, tgt).mean()
                    # loss2 = mse(x2, tgt).mean()
                    loss1 = reconstruction_loss(x1, tgt, loss_type=loss_type).mean()
                    loss2 = reconstruction_loss(x2, tgt, loss_type=loss_type).mean()
                    loss = (1 / n) * loss1 + (1 - 1 / n) * loss2
                else:
                    x_pred = out
                    loss = reconstruction_loss(x_pred, tgt, loss_type=loss_type).mean()

                    # backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1
            # test data AUROC compute
            #mean_auc = computeLoss(model, data, testD, train_dict_D, trainO, optimizer, scheduler, cfg)

            scheduler.step()
            avg_loss = total_loss / max(1, count)
            tqdm.write(f"Epoch {epoch},\tL1 = {total_loss / max(1, count)}")
            return avg_loss, optimizer.param_groups[0]["lr"]
        else:
            reduce_mode = cfg["scoring"]["reduce"]
            k_ratio = cfg["scoring"]["k_ratio"]
            car_p = cfg["scoring"]["car_positive_ratio"]
            batch_size = cfg["training"]["batch_size"]
            # Changed mainly
            car_scores = {}
            car_reconstruction_data = {}
            for cid, arr in tqdm(data.items()):
                scores = []
                result_data=[]
                arr = torch.as_tensor(arr, dtype=torch.float32, device=device)
                dataloader = DataLoader(arr, batch_size=batch_size, shuffle=False,num_workers=0)
                for batch in dataloader:
                    score,result = snippet_score(
                        model,
                        batch,
                        cfg=cfg,
                        device=device,
                        reduce=reduce_mode,
                        k_ratio=k_ratio,
                        p=car_p,
                    )
                    result_data.extend(result)
                    scores.extend(score.detach().cpu().tolist())
                car_scores[cid] = scores
                car_reconstruction_data[cid] = result_data
            return car_scores,car_reconstruction_data


def computeLoss(model, data, testD, train_dict_D, trainO, optimizer, scheduler, cfg):
    print(
        f"{color.HEADER}Testing {cfg['model']['name']} on {cfg['data']['dataset']}{color.ENDC}"
    )
    scores = backprop(
        0,
        model,
        testD,
        data,
        train_dict_D,
        trainO,
        optimizer,
        scheduler,
        cfg,
        training=False,
    )
    print("Calculate Training Data Score")
    train_scores = backprop(
        0,
        model,
        train_dict_D,
        data,
        testD,
        trainO,
        optimizer,
        scheduler,
        cfg,
        training=False,
    )
    mean_auc = fiveFold_AUROC(train_scores, scores, train_labels, labels)
    return mean_auc


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

    train_loader, test_loader, labels, train_dict, train_labels = load_dataset(
        cfg["data"]["output_folder"] + args.dataset, args.test
    )
    # Batch Size = Entire Time Series Data (L) 전체 데이터를 받아온다.

    # Added
    trainD = next(iter(train_loader))
    trainO = trainD
    testD = test_loader
    testO = testD
    print("trainD device:", getattr(trainD, "device", None))

    print(trainO.shape)
    print(type(testO))
    print(type(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dims = len(cfg["data"]["features"])


    model, optimizer, scheduler, epoch, accuracy_list = load_model(
        cfg["model"]["name"],
        device,
        dims,
        args,
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
                e,
                model,
                s,
                testD,
                train_dict,
                trainO,
                optimizer,
                scheduler,
                cfg,
                training=True,
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

        save_model(model, optimizer, scheduler, e, accuracy_list, args)
        # plot_accuracies(accuracy_list, f"{args.model}_{args.dataset}")

    ### Testing phase

    print(
        f"{color.HEADER}Testing {cfg['model']['name']} on {cfg['data']['dataset']}{color.ENDC}"
    )
    # loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    # Added
    scores, test_rec_data = backprop(
        0,
        model,
        testD,
        trainD,
        train_dict,
        trainO,
        optimizer,
        scheduler,
        training=False,
        cfg=cfg,
    )
    print("Calculate Training Data Score")
    train_scores, train_rec_data = backprop(
        0,
        model,
        train_dict,
        testD,
        trainD,
        trainO,
        optimizer,
        scheduler,
        cfg,
        training=False,
    )

    print("Save Score Files")
    config_path = Path(args.config)
    run_id = config_path.stem  # final path

    out_dir = Path("./results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "test"
    np.save(out_dir / f"{prefix}_scores.npy", scores, allow_pickle=True)
    torch.save(test_rec_data, out_dir / f"{prefix}_rec_data.pt")
    np.save(out_dir / "train_scores.npy", train_scores, allow_pickle=True)
    torch.save(train_rec_data, out_dir / "train_rec_data.pt")

    print("Save Score files Finished.")

    mean_auc = fiveFold_AUROC(train_scores, scores, train_labels, labels)
    print(f"AUROC={mean_auc:.6f}")
