from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def print_classification_report(labels_preds_kfold):
    for idx, data in enumerate(labels_preds_kfold):
        print(f"K-Fold: {idx}")
        print(classification_report(data[0], data[1], target_names=EMO_CLASSES.keys()))

def plot_metrics(data_kfold):
    num_folds = len(data_kfold)
    plt.figure(figsize=(20, 5))

    for idx, data in enumerate(data_kfold):
        acc, f1, recall, precision = data
        plt.subplot(1, num_folds, idx + 1)
        plt.plot(acc, label='Accuracy')
        plt.plot(recall, label='Recall')
        plt.plot(f1, label='F1')
        plt.plot(precision, label='Precision')
        plt.title(f"K-Fold {idx + 1}")
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# average classification report
def average_classification_report(labels_preds_kfold):
    all_labels = []
    all_preds = []
    for data in labels_preds_kfold:
        all_labels.extend(data[0])
        all_preds.extend(data[1])
    print(classification_report(all_labels, all_preds, target_names=EMO_CLASSES.keys()))

def plot_confusion_matrix(labels_preds_kfold):
    all_labels = []
    all_preds = []
    for data in labels_preds_kfold:
        all_labels.extend(data[0])
        all_preds.extend(data[1])

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMO_CLASSES.keys(), yticklabels=EMO_CLASSES.keys())
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def compute_ua_wa(labels_preds_kfold):
    true_labels = np.array([])
    predicted_labels = np.array([])

    # Sum all the true and predicted labels from all the k-folds
    for data in labels_preds_kfold:
        true_labels = np.append(true_labels, data[0])
        predicted_labels = np.append(predicted_labels, data[1])

    # Cal UA (Unweighted Accuracy)
    ua = accuracy_score(true_labels, predicted_labels)

    # Cal WA (Weighted Accuracy)
    wa = []
    for label in np.unique(true_labels):
        i_true = true_labels[true_labels == label]
        i_predicted = predicted_labels[true_labels == label]

        wa.append(np.sum(i_true == i_predicted) / len(i_true))
    wa = np.mean(wa)
    return ua, wa
