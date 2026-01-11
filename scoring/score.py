import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# 상위 몇 %를 잘라서 이상치로 보았을 때, 그 안의 precision이 가장 높은 구간
def find_best_percent(result, granularity_all=1000):
    """
    find threshold
    :param result: sorted result
    :param granularity_all: granularity_all
    """
    max_percent = 0
    best_n = 1
    print("threshold tuning start:")
    for n in tqdm(range(1, 100)):
        head_n = n / granularity_all
        data_length = max(round(len(result) * head_n), 1)
        count_dist = count_entries(result.loc[: data_length - 1], "label")
        try:
            percent = count_dist["1"] / (count_dist["0"] + count_dist["1"])
            # anormal 갯수 파악.
        except KeyError:
            print("can't find n%,take 1%")
            percent = 0.01
        if percent > max_percent:
            max_percent = percent
            best_n = n
    print("top %d / %s is the highest, %s" % (granularity_all, best_n, max_percent))
    print("Count dist : ", count_dist)
    return best_n, max_percent, granularity_all


def count_entries(df, col_name):
    """
    count
    """
    count_dist = {"0": 0, "1": 0}
    col = df[col_name]
    for entry in col:
        if str(int(entry)) in count_dist.keys():
            count_dist[str(int(entry))] = count_dist[str(int(entry))] + 1
        else:
            count_dist[str(int(entry))] = 1
    return count_dist


def find_best_result(
    threshold_n, result, dataframe_std, ind_car_num_list, ood_car_num_list
):
    """
    find_best_result
    :param threshold_n: threshold
    :param result: sorted result
    :param dataframe_std: label
    """
    best_result, best_h, best_re, best_fa, best_f1, best_precision = None, 0, 0, 0, 0, 0
    best_auroc = 0
    for h in tqdm(range(10, 1000, 5)):
        train_result = charge_to_car(threshold_n, result, head_n=h)
        f1, recall, false_rate, precision, accuracy, auroc = evaluation(
            dataframe_std, train_result, ind_car_num_list, ood_car_num_list
        )
        if auroc >= best_auroc:
            best_f1 = f1
            best_h = h
            best_re = recall
            best_precision = precision
            best_result = train_result
            best_auroc = auroc
    return best_result, best_h, best_re, best_precision, best_f1, best_auroc


def charge_to_car(threshold_n, rec_result, head_n=92):
    """
    mapping from charge to car
    :param threshold_n: threshold
    :param rec_result: sorted result
    :param head_n: top %n
    :param gran: granularity
    """
    gran = 1000
    result = []
    for grp in rec_result.groupby("car"):
        temp = grp[1].values[:, -1].astype(float)
        idx = max(round(head_n / gran * len(temp)), 1)
        error = np.mean(temp[:idx])
        result.append([grp[0], int(error > threshold_n), error, threshold_n])

        """ top_errors = temp[:idx]
        snip_pred = (top_errors > threshold_n).astype(int)

        ratio = snip_pred.mean() # head_n %의 구간 중 이상 스니펫 비율
        # result.append([grp[0], int(ratio > 0), error, threshold_n]) """
    return pd.DataFrame(result, columns=["car", "predict", "error", "threshold_n"])


def evaluation(dataframe_std, dataframe, ind_car_num_list, ood_car_num_list):
    """
    calculated statistics
    :param dataframe_std:
    :param dataframe:
    :return:
    """

    # calculate auroc
    #     print(dataframe) # error car
    _label = []
    for each_car in dataframe["car"]:
        if int(each_car) in ind_car_num_list:
            _label.append(0)
        if int(each_car) in ood_car_num_list:
            _label.append(1)

    fpr, tpr, thresholds = metrics.roc_curve(
        _label, list(dataframe["error"]), pos_label=1
    )
    auroc = auc(fpr, tpr)

    data = pd.merge(dataframe_std, dataframe, on="car")
    cm = confusion_matrix(data["label"].astype(int), data["predict"].astype(int))
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    false_rate = fp / (tn + fp) if tn + fp != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f1, recall, false_rate, precision, accuracy, auroc


def merge_scores(train_scores, test_scores, train_label_obj, test_label_obj):
    train_scores.update(test_scores)  # train, test 하나로 합침.
    ind_car_num_list = list(train_label_obj.keys())
    ood_car_num_list = []
    rows = []
    labels = []

    for k, v in test_label_obj.items():
        if test_label_obj[k] == 1:
            ood_car_num_list.append(k)
        else:
            ind_car_num_list.append(k)

    all_car_num_list = set(ind_car_num_list + ood_car_num_list)
    print("Number of Merged car : ", len(all_car_num_list))
    print("Number of Normal car : ", len(ind_car_num_list))
    print("Number of Abnormal car : ", len(ood_car_num_list))
    for each_car in all_car_num_list:
        if each_car in ind_car_num_list:
            label = 0
        elif each_car in ood_car_num_list:
            label = 1
        labels.append([each_car, int(label)])

    for car_id, snippet_scores in train_scores.items():
        if car_id in ind_car_num_list:
            label = 0
        elif car_id in ood_car_num_list:
            label = 1
        else:
            continue

        for s in snippet_scores:
            rows.append([car_id, label, float(s)])

    dataframe = pd.DataFrame(labels, columns=["car", "label"])
    all_snippet_df = pd.DataFrame(rows, columns=["car", "label", "rec_error"])

    return (
        all_snippet_df,
        dataframe,
        all_car_num_list,
        ind_car_num_list,
        ood_car_num_list,
    )


def fiveFold_AUROC(train_scores, scores, train_labels, labels):
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

    return mean_auc
