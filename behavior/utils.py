from datetime import datetime
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)


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


def save_data_prediction(save_path, label, pred, conf, data, ldts):
    """
    data: np.ndary
        Lx4: L: length is usually 20
    """
    rand = np.random.randint(0, 255, 1)[0]
    gps = np.float32(data[0, -1] * 22.3012351755624)
    t = datetime.utcfromtimestamp(ldts[2]).strftime("%Y-%m-%d %H:%M:%S.%f")
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
    plt.show(block=False)
    plt.savefig(fail_path / f"confusion_matrix_{stage}.png", bbox_inches="tight")
    plot_confusion_matrix(confmat, target_labels)
    plt.show(block=False)

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
