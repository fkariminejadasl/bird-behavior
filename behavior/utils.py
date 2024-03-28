from datetime import datetime
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

gps_scale = 22.3012351755624


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
    _, ax = plt.subplots(1, 1)
    ax.plot(data[:, 0], "r-*", data[:, 1], "b-*", data[:, 2], "g-*")
    ax.set_xlim(0, 20)
    ax.set_ylim(-3.5, 3.5)
    plt.show(block=False)
    return ax


def save_data_prediction(save_path, label, pred, conf, data, ldts):
    """
    data: np.ndary
        Lx4: L: length is usually 20
    """
    rand = np.random.randint(0, 255, 1)[0]
    gps = np.float32(data[0, -1] * gps_scale)
    t = datetime.utcfromtimestamp(ldts[2]).strftime("%Y-%m-%d %H:%M:%S")
    name = f"time:{t}, gps:{gps:.4f}, dev:{ldts[1]},\nlabel:{label}, pred:{pred}, conf:{conf:.1f}"
    _, ax = plt.subplots(1, 1)
    ax.plot(data[:, 0], "r-*", data[:, 1], "b-*", data[:, 2], "g-*")
    ax.set_xlim(0, 20)
    ax.set_ylim(-3.5, 3.5)
    plt.title(name)
    name = " ".join(name.split("\n"))
    plt.savefig(save_path / f"{name}_{rand}.png", bbox_inches="tight")
    plt.close()
    return name


def save_predictions_csv(
    save_file, data, idts, llat, mod_name, preds, probs, target_labels_names
):
    idts = np.array(idts)
    data = data.transpose(2, 1).cpu().numpy()
    with open(save_file, "w") as rfile:
        rfile.write(
            "device,time,index,GPS,prediction,confidence,latitude,longitude,altitude,temperature,runtime,model\n"
        )
        for i in range(len(data)):
            device_id = idts[i, 1]
            start_date = datetime.utcfromtimestamp(idts[i, 2]).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            index = idts[i, 0]
            gps = np.float32(data[i, 0, -1] * gps_scale)
            pred = preds[i]
            pred_name = target_labels_names[pred]
            conf = probs[i, pred]
            latitude, longitude, altitude, temperature = llat[i]
            runtime_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            text = (
                f"{device_id},{start_date},{index},{gps:.4f},{pred_name},{conf:.2f},"
                f"{latitude:.7f},{longitude:.7f},{int(altitude)},{temperature:.1f},"
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
    fail_path,
    target_labels,
    target_labels_names,
    n_classes,
    stage="valid",
    SAVE_FAILED=False,
):
    with torch.no_grad():
        labels = ldts[:, 0]
        labels = labels.to(device)
        data = data.to(device)  # N x C x L
        outputs = model(data)  # N x C
        prob = torch.nn.functional.softmax(outputs, dim=-1).detach()  # N x C
        pred = torch.argmax(outputs.data, 1)
        # loss and accuracy
        loss = criterion(outputs, labels)  # 1
        corrects = (pred == labels).sum().item()
        accuracy = corrects / len(labels) * 100

    labels = labels.cpu().numpy()
    prob = prob.cpu().numpy()
    pred = pred.cpu().numpy()

    # confusion matrix
    confmat = confusion_matrix(labels, pred, labels=np.arange(len(target_labels)))
    # cm_percentage = cm.astype('float') / confmat.sum(axis=1)[:, np.newaxis] * 100
    plot_confusion_matrix(confmat, target_labels_names)
    plt.savefig(fail_path / f"confusion_matrix_{stage}.png", bbox_inches="tight")
    plot_confusion_matrix(confmat, target_labels)

    # if one of the classes is empty
    inds = np.where(np.all(confmat == 0, axis=1) == True)[0]  # indices of zero rows
    if len(inds) != 0:
        labels = np.concatenate((labels, inds))
        prob = np.concatenate((prob, np.zeros((len(inds), prob.shape[1]))))

    if n_classes == 2:
        ap = average_precision_score(labels, np.argmax(prob, axis=1))
    else:
        ap = average_precision_score(labels, prob)
    print(ap, loss.item(), accuracy)
    print(confmat)

    if SAVE_FAILED:
        names = []
        inds = np.where(pred != labels)[0]
        for ind in inds:
            label_name = target_labels_names[labels[ind]]
            if label_name == "Pecking":
                pred_name = target_labels_names[pred[ind]]
                conf = prob[ind, pred[ind]]
                data_item = data[ind].transpose(1, 0).cpu().numpy()
                ldts_item = ldts[ind]
                name = save_data_prediction(
                    fail_path, label_name, pred_name, conf, data_item, ldts_item
                )
                names.append(name)
        with open(fail_path / "results.txt", "a") as f:
            [f.write(f"{name}\n") for name in names]


def save_results(
    data,
    idts,
    llat,
    model_checkpoint,
    model,
    device,
    fail_path,
    target_labels_names,
):
    data = data.to(device)  # N x C x L
    outputs = model(data)  # N x C
    probs = torch.nn.functional.softmax(outputs, dim=-1).detach()  # N x C
    preds = torch.argmax(outputs.data, 1)
    probs = probs.cpu().numpy()
    preds = preds.cpu().numpy()

    save_predictions_csv(
        fail_path / "results.csv",
        data,
        idts,
        llat,
        model_checkpoint,
        preds,
        probs,
        target_labels_names,
    )

    # TODO saving some data
    ind = 0
    pred_name = target_labels_names[preds[ind]]
    conf = probs[ind, preds[ind]]
    data_item = data[ind].transpose(1, 0).cpu().numpy()
    ldts_item = idts[ind]
    _ = save_data_prediction(
        fail_path, pred_name, pred_name, conf, data_item, ldts_item
    )
