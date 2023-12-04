import matplotlib.pylab as plt
import numpy as np


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
