from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

gps_scale = 22.3012351755624

ind2name = {
    0: "Flap",
    1: "ExFlap",
    2: "Soar",
    3: "Boat",
    4: "Float",
    5: "SitStand",
    6: "TerLoco",
    7: "Other",
    8: "Manouvre",
    9: "Pecking",
}

target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # no Other
target_labels_names = [ind2name[t] for t in target_labels]
new_label_inds = np.arange(len(target_labels))
n_classes = len(target_labels_names)
# target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# target_labels = [0, 2, 3, 4, 5, 6] # no: Exflap:1, Other:7, Manauvre:8, Pecking:9
# target_labels = [0, 3, 4, 5, 6]  # no: Exflap:1, Soar:2, Other:7, Manauvre:8, Pecking:9
# target_labels = [0, 2, 4, 5]
# target_labels = [8, 9]
# target_labels = [0, 1, 2, 3, 4, 5, 6, 9]  # no Other:7; combine soar:2 and manuver:8


def save_gimus_idts(save_file, gimus, idts):
    wfile = open(save_file, "w")
    for gimu, idt in zip(gimus, idts):
        ind = idt[0]
        device_id = idt[1]
        date = datetime.fromtimestamp(idt[2], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        gimus = np.round(gimus, 6)
        line = f"{device_id},{date},{ind},{-1},{gimu[0]:.6f},{gimu[1]:.6f},{gimu[2]:.6f},{gimu[3]:.6f}\n"
        wfile.write(line)
    wfile.close()


def compare_rows(row1, row2, subset):
    row1 = row1[subset].reset_index(drop=True)
    row2 = row2[subset].reset_index(drop=True)
    return row1.equals(row2)


# matches = df.apply(lambda row: compare_rows(row, query_row), axis=1)
# matches = dow[subset].eq(query[subset]).all(axis=1)
# dow.index[matches]
def set_seed(seed):
    """Ensure reproducibility."""
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multiple gpu
    # generator = torch.Generator().manual_seed(seed)  # for random_split
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title("Confusion Matrix")
    plt.colorbar()

    num_classes = len(class_names)
    plt.xticks(np.arange(num_classes), class_names, rotation=45)
    plt.yticks(np.arange(num_classes), class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                str(confusion_matrix[i, j]),
                horizontalalignment="center",
                color="black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


def plot_one(data):
    data_len = data.shape[0]  # 20, 60, 200
    _, ax = plt.subplots(1, 1)
    ax.plot(data[:, 0], "r-*", data[:, 1], "b-*", data[:, 2], "g-*")
    ax.set_xlim(0, data_len)
    ax.set_ylim(-3.5, 3.5)
    plt.xticks(np.linspace(0, data_len, 5).astype(np.int64))
    plt.title(f"gps speed: {data[0,-1]:.2f}")
    plt.show(block=False)
    return ax


# s = df_s.groupby(by=[0,1]).get_group((533, "2012-05-15 03:10:11")).sort_values(by=[2])
def plot_all(dataframe, glen=20):
    n_plots = len(dataframe) // glen
    fig, axs = plt.subplots(1, n_plots, figsize=(18, 4))
    for i, ax in enumerate(axs):
        slice = dataframe.iloc[i * glen : i * glen + glen]
        data = slice[[4, 5, 6]].values
        indices = slice[[2]].values.squeeze()
        ax.plot(
            indices,
            data[:, 0],
            "r-*",
            indices,
            data[:, 1],
            "b-*",
            indices,
            data[:, 2],
            "g-*",
        )
        ax.set_xlim(indices[0], indices[-1])
        ax.set_ylim(-3.5, 3.5)
        ax.set_xticks(np.linspace(indices[0], indices[-1], 5).astype(np.int64))
        label = ind2name[slice.iloc[0, 3]]
        ax.set_title(f"label: {label}")
    fig.suptitle(f"gps: {slice.iloc[0,7]:.2f}")  # , fontsize=16
    plt.tight_layout()
    plt.show(block=False)
    return ax


def save_data_prediction(save_path: Path, label, pred, conf, data, ldts, idx):
    """
    data: np.ndary
        Lx4: L: length is usually 20
    """
    save_path.mkdir(parents=True, exist_ok=True)
    gps = np.float32(data[0, -1] * gps_scale)
    t = datetime.fromtimestamp(ldts[2], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    name = f"{idx}, time:{t}, dev:{ldts[1]}, gps:{gps:.4f},\nlabel:{label}, pred:{pred}, conf:{conf:.1f}"
    _, ax = plt.subplots(1, 1)
    ax.plot(data[:, 0], "r-*", data[:, 1], "b-*", data[:, 2], "g-*")
    ax.set_xlim(0, 20)
    ax.set_ylim(-3.5, 3.5)
    plt.title(name)
    name = " ".join(name.split("\n"))
    plt.savefig(save_path / f"{name}.png", bbox_inches="tight")
    plt.close()
    return name


def save_predictions_csv(
    save_file, data, idts, llats, mod_name, preds, probs, target_labels_names
):
    idts = np.array(idts)
    data = data.transpose(2, 1).cpu().numpy()
    with open(save_file, "w") as rfile:
        rfile.write(
            "device_info_serial,date_time,index,speed_2d,prediction,confidence,latitude,longitude,altitude,temperature,runtime,model\n"
        )
        for i in range(len(data)):
            device_id = idts[i, 1]
            start_date = datetime.fromtimestamp(idts[i, 2], tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            index = idts[i, 0]
            gps = np.float32(data[i, 0, -1] * gps_scale)
            pred = preds[i]
            pred_name = target_labels_names[pred]
            conf = probs[i, pred]
            runtime_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if llats is not None:
                latitude, longitude, altitude, temperature = llats[i]
                text = (
                    f"{device_id},{start_date},{index},{gps:.4f},{pred_name},{conf:.2f},"
                    f"{latitude:.7f},{longitude:.7f},{int(altitude)},{temperature:.1f},"
                    f"{runtime_date},{mod_name}\n"
                )
            else:
                text = (
                    f"{device_id},{start_date},{index},{gps:.4f},{pred_name},{conf:.2f},"
                    "None,None,None,None,"
                    f"{runtime_date},{mod_name}\n"
                )
            rfile.write(text)


def load_predictions_csv(save_file):
    device_ids = []
    dates = []
    indices = []
    gpss = []
    pred_texts = []
    confs = []
    with open(save_file, "r") as wfile:
        _ = wfile.readline()
        for row in wfile:
            items = row.strip().split(",")
            device_ids.append(int(items[0]))
            dates.append(items[1])
            indices.append(int(items[2]))
            gpss.append(float(items[3]))
            pred_texts.append(items[4])
            confs.append(float(items[5]))
    return device_ids, dates, indices, gpss, pred_texts, confs


def precision_recall(
    labels: Iterable[float], predictions: Iterable[float]
) -> Tuple[float, float, float]:
    """Compute precision, recall and f-score from given labels and predictions"""
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (recall * precision) / (recall + precision) * 100
    return precision, recall, f_score


def helper_results(
    data,
    ldts,
    model,
    criterion,
    device,
    fail_path: Path,
    target_labels_names,
    n_classes,
    stage="valid",
    SAVE_FAILED=False,
):
    with torch.no_grad():
        labels = ldts[:, 0]
        labels = labels.to(device)  # N
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        prob = torch.nn.functional.softmax(outputs, dim=-1).detach()  # N x C
        pred = torch.argmax(outputs.data, 1)  # N
        # loss and accuracy
        loss = criterion(outputs, labels)  # 1
        corrects = (pred == labels).sum().item()
        accuracy = corrects / len(labels) * 100

    labels = labels.cpu().numpy()
    prob = prob.cpu().numpy()
    pred = pred.cpu().numpy()

    # confusion matrix
    confmat = confusion_matrix(labels, pred, labels=np.arange(len(target_labels_names)))
    # cm_percentage = cm.astype('float') / confmat.sum(axis=1)[:, np.newaxis] * 100
    plot_confusion_matrix(confmat, target_labels_names)
    plt.savefig(fail_path / f"confusion_matrix_{stage}.png", bbox_inches="tight")

    metrics_df = per_class_statistics(confmat, target_labels_names)
    metrics_df.to_csv(fail_path / f"per_class_metrics_{stage}.csv", index=False)
    metrics_df = per_class_statistics_balanced(confmat, target_labels_names)
    metrics_df.to_csv(
        fail_path / f"per_class_metrics_balanced_{stage}.csv", index=False
    )

    # if one of the classes is empty
    inds = np.where(np.all(confmat == 0, axis=1) == True)[0]  # indices of zero rows
    if len(inds) != 0:
        labels = np.concatenate((labels, inds))
        prob = np.concatenate((prob, np.zeros((len(inds), prob.shape[1]))))

    if n_classes == 2:
        ap = average_precision_score(labels, np.argmax(prob, axis=1))
    else:
        ap = average_precision_score(labels, prob)
    print(f"AP: {ap:.2f}, Loss: {loss.item():.2f}, Accuracy: {accuracy:.2f}")
    print(confmat)

    if SAVE_FAILED:
        names = []
        inds = np.where(pred != labels)[0]
        for i, ind in enumerate(inds):
            label_name = target_labels_names[labels[ind]]
            if label_name == "Pecking":
                pred_name = target_labels_names[pred[ind]]
                conf = prob[ind, pred[ind]]
                data_item = data[ind].transpose(1, 0).cpu().numpy()
                ldts_item = ldts[ind]
                name = save_data_prediction(
                    fail_path, label_name, pred_name, conf, data_item, ldts_item, i
                )
                names.append(name)
        with open(fail_path / "results.txt", "a") as f:
            [f.write(f"{name}\n") for name in names]


def save_plots_for_specific_label(
    data,
    ldts,
    model,
    device,
    fail_path: Path,
    target_labels_names,
    query_label="Pecking",
):
    with torch.no_grad():
        labels = ldts[:, 0]
        labels = labels.to(device)  # N
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        prob = torch.nn.functional.softmax(outputs, dim=-1).detach()  # N x C
        pred = torch.argmax(outputs.data, 1)  # N

    labels = labels.cpu().numpy()
    prob = prob.cpu().numpy()
    pred = pred.cpu().numpy()

    query_ind = [
        int(ind)
        for ind, name in zip(new_label_inds, target_labels_names)
        if name == query_label
    ][0]
    inds = np.where(labels == query_ind)[0]
    save_path = fail_path / query_label
    for i, ind in enumerate(inds):
        label_name = target_labels_names[labels[ind]]
        pred_name = target_labels_names[pred[ind]]
        conf = prob[ind, pred[ind]]
        data_item = data[ind].transpose(1, 0).cpu().numpy()
        ldts_item = ldts[ind]
        _ = save_data_prediction(
            save_path, label_name, pred_name, conf, data_item, ldts_item, i
        )


def save_results(
    save_file,
    data,
    idts,
    llats,
    model_checkpoint,
    model,
    device,
    target_labels_names,
):
    data = data.to(device)  # N x C x L
    with torch.no_grad():
        outputs = model(data)  # N x C
    probs = torch.nn.functional.softmax(outputs, dim=-1).detach()  # N x C
    preds = torch.argmax(outputs.data, 1)
    probs = probs.cpu().numpy()
    preds = preds.cpu().numpy()

    save_predictions_csv(
        save_file,
        data,
        idts,
        llats,
        model_checkpoint,
        preds,
        probs,
        target_labels_names,
    )


def per_class_statistics(conf_matrix, target_labels_names):
    # Calculating True Positives, False Positives, False Negatives, and True Negatives for each class
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    # Calculating precision, recall, F1 score for each class
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Create a DataFrame for easy display
    metrics_df = pd.DataFrame(
        {
            "Class": target_labels_names,  # range(num_classes),
            "True Positives (TP)": TP,
            "False Positives (FP)": FP,
            "False Negatives (FN)": FN,
            "Precision": np.round(precision, 2),
            "Recall": np.round(recall, 2),
            "F1 Score": np.round(f1_score, 2),
        }
    )

    # Calculate combined metrics
    total_TP = np.sum(TP)
    total_FP = np.sum(FP)
    total_FN = np.sum(FN)

    # Calculating combined precision, recall, F1 score
    combined_precision = total_TP / (total_TP + total_FP)
    combined_recall = total_TP / (total_TP + total_FN)
    combined_f1_score = (
        2
        * (combined_precision * combined_recall)
        / (combined_precision + combined_recall)
    )

    # Append combined stats to the DataFrame
    combined_metrics = pd.DataFrame(
        {
            "Class": ["Combined"],
            "True Positives (TP)": [total_TP],
            "False Positives (FP)": [total_FP],
            "False Negatives (FN)": [total_FN],
            "Precision": [np.round(combined_precision, 2)],
            "Recall": [np.round(combined_recall, 2)],
            "F1 Score": [np.round(combined_f1_score, 2)],
        }
    )

    # Combine per class metrics with combined metrics
    full_metrics_df = pd.concat([metrics_df, combined_metrics], ignore_index=True)

    return full_metrics_df


def per_class_statistics_balanced(conf_matrix, target_labels_names):
    # Step 1: Calculate class weights based on inverse class frequencies
    num_classes = conf_matrix.shape[0]
    class_totals = np.sum(conf_matrix, axis=1)
    total_samples = np.sum(class_totals)
    class_weights = total_samples / (num_classes * class_totals)

    # Step 2: Create a weighted confusion matrix by applying the class weights
    weighted_conf_matrix = conf_matrix * class_weights[:, np.newaxis]

    # Step 3: Recalculate True Positives, False Positives, False Negatives, and True Negatives for each class
    TP_weighted = np.diag(weighted_conf_matrix)
    FP_weighted = np.sum(weighted_conf_matrix, axis=0) - TP_weighted
    FN_weighted = np.sum(weighted_conf_matrix, axis=1) - TP_weighted

    # Step 4: Recalculate precision, recall, F1 score based on the weighted confusion matrix
    precision_weighted = TP_weighted / (TP_weighted + FP_weighted)
    recall_weighted = TP_weighted / (TP_weighted + FN_weighted)
    f1_score_weighted = (
        2
        * (precision_weighted * recall_weighted)
        / (precision_weighted + recall_weighted)
    )

    # Create a DataFrame for per-class weighted metrics
    weighted_metrics_df = pd.DataFrame(
        {
            "Class": target_labels_names,
            "True Positives (TP)": np.int64(np.round(TP_weighted)),
            "False Positives (FP)": np.int64(np.round(FP_weighted)),
            "False Negatives (FN)": np.int64(np.round(FN_weighted)),
            "Precision": np.round(precision_weighted, 2),
            "Recall": np.round(recall_weighted, 2),
            "F1 Score": np.round(f1_score_weighted, 2),
        }
    )

    # Step 5: Calculate combined statistics
    total_TP_weighted = np.sum(TP_weighted)
    total_FP_weighted = np.sum(FP_weighted)
    total_FN_weighted = np.sum(FN_weighted)

    # Calculate combined precision, recall, and F1 score
    combined_precision_weighted = total_TP_weighted / (
        total_TP_weighted + total_FP_weighted
    )
    combined_recall_weighted = total_TP_weighted / (
        total_TP_weighted + total_FN_weighted
    )
    combined_f1_score_weighted = (
        2
        * (combined_precision_weighted * combined_recall_weighted)
        / (combined_precision_weighted + combined_recall_weighted)
    )

    # Create a row for the combined statistics
    combined_metrics = pd.DataFrame(
        {
            "Class": ["Combined"],
            "True Positives (TP)": [np.int64(np.round(total_TP_weighted))],
            "False Positives (FP)": [np.int64(np.round(total_FP_weighted))],
            "False Negatives (FN)": [np.int64(np.round(total_FN_weighted))],
            "Precision": [np.round(combined_precision_weighted, 2)],
            "Recall": [np.round(combined_recall_weighted, 2)],
            "F1 Score": [np.round(combined_f1_score_weighted, 2)],
        }
    )

    # Combine per-class metrics with combined metrics
    full_metrics_df = pd.concat(
        [weighted_metrics_df, combined_metrics], ignore_index=True
    )

    return full_metrics_df


"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# Confusion matrix provided by the user
conf_matrix = np.array([
    [ 565,    1,    0,    0,    0,    0,    0,    2,    0],
    [   0,   36,    0,    0,    0,    0,    0,    2,    0],
    [   0,    0,  470,    2,    4,    0,    0,   14,    0],
    [   0,    0,    2,  159,    0,    0,    0,    0,    1],
    [   0,    0,    6,    0,  541,    0,    0,    0,    0],
    [   0,    1,    7,    0,    4, 1332,    4,    0,   19],
    [   0,    0,    2,    0,    0,    2,  293,    0,    5],
    [   0,    0,   15,    0,    0,    0,    0,  125,    1],
    [   0,    0,    4,    1,    0,   18,   21,    2,  267]
])

metrics_df = per_class_statistics(conf_matrix, target_labels_names)
print(metrics_df)
# metrics_df.to_csv("/home/fatemeh/Downloads/per_class_metrics.csv", index=False)
metrics_df = per_class_statistics_balanced(conf_matrix, target_labels_names)
print(metrics_df)
# metrics_df.to_csv("/home/fatemeh/Downloads/per_class_metrics_balanced.csv", index=False)
"""

# ind = 0
# pred_name = target_labels_names[preds[ind]]
# conf = probs[ind, preds[ind]]
# data_item = data[ind].transpose(1, 0).cpu().numpy()
# ldts_item = idts[ind]
# _ = save_data_prediction(
#     save_file.parent, 'float', 'pred_float', .54, data_item, ldts_item, i
# )
