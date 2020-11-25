import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns;

sns.set()
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.calibration import calibration_curve


# Plots for model validation and diagnosis.
def plot_metrics(history):
    metrics = ['loss', 'AUC', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.6, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    AUROC = auc(fp, tp)
    print('AUROC: ', AUROC)
    # plt.xlim([-0.5,40])
    # plt.ylim([60,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    return AUROC


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    asc = accuracy_score(labels, predictions > p)
    print('Accuracy: ', asc)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    sensibility = cm[1][1]/cm[1, :].sum()
    specificity = cm[0][0]/cm[0, :].sum()
    PPV = cm[1][1]/cm[:, 1].sum()
    NPV = cm[0][0]/cm[:, 0].sum()

    print('Sensibility: ', sensibility)
    print('Specificity: ', specificity)
    print('Positive Predictive Value: ', PPV)
    print('Negative Predictive Value: ', NPV)
    return sensibility, specificity, PPV, NPV


def plot_pr_curve(name, labels, predictions, **kwargs):
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    AUPRC = auc(recall, precision)
    print('AUPRC score', AUPRC)
    plt.plot(100 * recall, 100 * precision, label=name, linewidth=2, **kwargs)
    plt.xlabel('recall [%]')
    plt.ylabel('precision [%]')
    # plt.xlim([-0.5,40])
    # plt.ylim([60,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.legend()
    ax.set_aspect('equal')
    return AUPRC


def plot_calibration(name, labels, predictions, **kwargs):
    y, x = calibration_curve(labels, predictions, n_bins=20)
    # calibration curves
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot(x, y, "s-", label="%s" % (name,))
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    # transform = ax.transAxes
    # line.set_transform(transform)
    ax1.add_line(line)
    ax2.hist(predictions, range=(0, 1), bins=20, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.plot(y, x, marker='o', linewidth=1, label=name)
