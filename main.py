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


def backprop(epoch, model, data, dataO, optimizer, scheduler, cfg, training=True):
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

            batch_size = cfg["training"]["batch_size"]
            loss_type = cfg["training"]["loss_type"]
            dataloader = DataLoader(data_x, shuffle=True, batch_size=batch_size)
            count = 0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                batch = batch.to(device, non_blocking=True)

                batch = convert_to_windows_mod(batch, cfg, model)

                B, N_win, L, F = (
                    batch.shape
                )  # Batch size, Number of window, window length, Feature

                if count == 0:
                    print("batch.shape:", batch.shape, " w_size:", w_size)
                    count += 1

                src = (
                    batch.permute(2, 0, 1, 3).contiguous().view(w_size, -1, F)
                )  # (Window,128,8)

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
                    if n<2:
                        loss = (1 / n) * loss1 + (1 - 1 / n) * loss2
                    else :
                        loss = loss2
                else:
                    x_pred = out
                    loss = reconstruction_loss(x_pred, tgt, loss_type=loss_type).mean()

                    # backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            scheduler.step()
            avg_loss = total_loss / max(1, count)
            tqdm.write(f"Epoch {epoch},\tL1 = {total_loss / max(1, count)}")
            return avg_loss, optimizer.param_groups[0]["lr"]
        else:
            reduce_mode = cfg["scoring"]["reduce"]
            k_ratio = cfg["scoring"]["k_ratio"]
            car_p = cfg["scoring"]["car_positive_ratio"]
            
            # Changed mainly
            car_scores = {}
            car_reconstruction_data = {}
            for cid, arr in tqdm(data.items()):
                scores = []
                result_data = []
                arr = torch.as_tensor(arr[:, :, :])
                for i in range(arr.shape[0]):
                    score,result = snippet_score(
                        model,
                        arr[i],
                        cfg=cfg,
                        reduce=reduce_mode,
                        k_ratio=k_ratio,
                        p=car_p,
                    )
                    result_data.append(result)
                    scores.append(score)
                car_scores[cid] = scores
                car_reconstruction_data[cid] = result_data
            return car_scores, car_reconstruction_data


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

    print(trainO.shape)
    print(type(testO))
    print(type(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, optimizer, scheduler, epoch, accuracy_list = load_model(
        cfg["model"]["name"],
        device,
        trainO.shape[-1],
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

        save_model(model, optimizer, scheduler, e, accuracy_list, args)
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
        scores,test_rec_data = backprop(
            0, model, testD, trainO, optimizer, scheduler, training=False, cfg=cfg
        )
        print("Calculate Training Data Score")
        train_scores,train_rec_data = backprop(
            0, model, train_dict, trainO, optimizer, scheduler, cfg, training=False
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
    
    # AUROC scoring
    print("AUROC Scoring by using DyAD method")
    all_snippet_df, dataframe, all_car_num_list, ind_car_num_list, ood_car_num_list = (
        merge_scores(train_scores, scores, train_labels, labels)
    )

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    AUC_fivefold_list = []

    for i in range(5):

        fold_num = i
        test_car_list = (
            ind_car_num_list[
                int(fold_num * len(ind_car_num_list) / 5) : int(
                    (fold_num + 1) * len(ind_car_num_list) / 5
                )
            ]
            + ood_car_num_list[: int(fold_num * len(ood_car_num_list) / 5)]
            + ood_car_num_list[int((fold_num + 1) * len(ood_car_num_list) / 5) :]
        )
        test_car_list = set(test_car_list)
        train_car_list = set(all_car_num_list - test_car_list)

        # ------------------------------
        # Train part: threshold_n, best_h 튜닝
        # ------------------------------
        train_result = all_snippet_df[all_snippet_df["car"].isin(train_car_list)].copy()
        test_result = all_snippet_df[all_snippet_df["car"].isin(test_car_list)].copy()

        train_res_csv = train_result[["label", "car", "rec_error"]].to_numpy()
        test_res_csv = test_result[["label", "car", "rec_error"]].to_numpy()

        rec_sorted_index = np.argsort(
            -train_res_csv[:, 2].astype(float)
        )  # 정렬한 인덱스 반환
        res = [
            train_res_csv[j][[1, 0, 2]] for j in rec_sorted_index
        ]  # [car, label, rec_error]
        result = pd.DataFrame(res, columns=["car", "label", "rec_error"])

        best_n, max_percent, granularity = find_best_percent(
            result, granularity_all=1000
        )
        head_n = best_n / granularity
        data_length = round(len(result) * head_n)
        # threshold_n : precision이 최대가 되는 지점의 임계값.
        threshold_n = result["rec_error"].values[data_length - 1].astype(float)

        print("threshold_n", threshold_n)
        print("start tuning, flag is", "rec_error")
        best_result, best_h, best_re, best_fa, best_f1, best_auroc = find_best_result(
            threshold_n, result, dataframe, ind_car_num_list, ood_car_num_list
        )
        if dataframe.shape[0] != best_result.shape[0]:
            print(
                "dataframe_std is ",
                dataframe.shape[0],
                "&&   dataframe is ",
                best_result.shape[0],
            )

        print("F1 Scores through Train data")
        print("best 1000 / %d:" % best_h)
        print("re:", best_re)
        print("fa:", best_fa)
        print("F1:", best_f1)

        # ------------------------------
        # Test part: charge_to_car → car-level score / 예측
        # ------------------------------
        rec_sorted_index = np.argsort(-test_res_csv[:, 2].astype(float))
        res = [test_res_csv[j][[1, 0, 2]] for j in rec_sorted_index]
        result = pd.DataFrame(res, columns=["car", "label", "rec_error"])
        result["car"] = result["car"].astype("int").astype("str")

        test_result_car = charge_to_car(threshold_n, result, head_n=best_h)
        # columns: ['car', 'predict', 'error', 'threshold_n']

        _score = list(test_result_car["error"])
        y_true = []
        for each_car in test_result_car["car"]:
            if int(each_car) in ind_car_num_list:
                y_true.append(0)
            if int(each_car) in ood_car_num_list:
                y_true.append(1)
        y_pred = list(
            test_result_car["predict"]
        )  # charge_to_car에서 0/1로 만들어 둔 것

        print("len(_score)", len(_score))
        fpr, tpr, thresholds = metrics.roc_curve(y_true, _score, pos_label=1)
        auc_fold = auc(fpr, tpr)
        print("AUC", auc_fold)
        AUC_fivefold_list.append(auc_fold)
        # np.save(f"/results/true_score_fold{i}.npy",y_true)
        # np.save(f"/results/pred_score_fold{i}.npy",_score)

    # ------------------------------
    # 5-fold 평균 ROC + 표준편차 밴드
    # ------------------------------

    print("Fold AUCs:", AUC_fivefold_list)
    mean_auc = np.mean(AUC_fivefold_list)
    print("AUC mean ", mean_auc)
    print(f"AUROC={mean_auc:.6f}")
