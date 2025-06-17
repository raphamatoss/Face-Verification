import matplotlib.pyplot as plt
import pipelines as p
from pathlib import Path
from tqdm import tqdm
import csv

def evaluate(dataset, dataset_path, cosine_threshold):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative= 0
    total = 0

    match_pairs = Path(dataset_path).joinpath("matchpairsDevTest.csv")
    mismatch_pairs = Path(dataset_path).joinpath("mismatchpairsDevTest.csv")

    with open(match_pairs, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in tqdm(reader):
            if len(row) < 3:
                continue
            name = row[0].strip()
            img1 = int(row[1].strip())
            img2 = int(row[2].strip())

            feat1 = dataset[name][img1-1]
            feat2 = dataset[name][img2-1]

            cosine_similarity = p.cos_similarity(feat1, feat2)

            if cosine_similarity >= cosine_threshold:
                true_positive += 1
            else:
                false_negative += 1
            total += 1

    with open(mismatch_pairs, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in tqdm(reader):
            if len(row) < 4:
                continue
            name1 = row[0].strip()
            img1 = int(row[1].strip())
            name2 = row[2].strip()
            img2 = int(row[3].strip())

            feat1 = dataset[name1][img1 - 1]
            feat2 = dataset[name2][img2 - 1]

            cosine_similarity = p.cos_similarity(feat1, feat2)

            if cosine_similarity < cosine_threshold:
                true_negative += 1
            else:
                false_positive += 1
            total += 1

    return true_positive, false_positive, true_negative, false_negative, total

def multi_eval(dataset, path, thresholds):
    accuracy_list = []
    recall_list = []
    f_score_list = []
    for threshold in thresholds:
        tp, fp, tn, fn, t = evaluate(dataset, path, threshold)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        recall = tp / (tp + fn)
        f_score = 2 * (recall * accuracy) / (recall + accuracy)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        f_score_list.append(f_score)

    plot_metrics(accuracy_list, recall_list, f_score_list, thresholds)

def plot_metrics(accuracy, recall, f_score, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].plot(thresholds, accuracy)
    axes[0].set_title("Accuracy over thresholds")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Threshold")

    axes[1].plot(thresholds, recall)
    axes[1].set_title("Recall over thresholds")
    axes[1].set_ylabel("Recall")
    axes[1].set_xlabel("Threshold")

    axes[2].plot(thresholds, f_score)
    axes[2].set_title("F1-Score over thresholds")
    axes[2].set_ylabel("F1-Score")
    axes[2].set_xlabel("Threshold")

    plt.show()
