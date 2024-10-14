from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, KFold

from utils import *
from common import *
from load_data import *
from model import *

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# import pad sequence

avarage = 'micro'

data_folder = "../VNEMOS/"

mfccs, labels = load_data(data_folder)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

EMO_CLASSES = {label: i for i, label in enumerate(np.unique(labels))}

for idx, (train_idx, test_idx) in enumerate(skf.split(mfccs, labels)):

    # wave_train, wave_test = waves[train_idx], waves[test_idx]
    mfcc_train, mfcc_test = mfccs[train_idx], mfccs[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    train_labels = [EMO_CLASSES[label] for label in train_labels]
    test_labels = [EMO_CLASSES[label] for label in test_labels]


    train_dataset = EmoDataset(mfcc_train, train_labels)
    test_dataset = EmoDataset(mfcc_test, test_labels)

    # print("Train data: ", Counter(train_labels))
    # print("Test data: ", Counter(test_labels))

epochs = 100
batch_size = 32
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMO_CLASSES = {label: i for i, label in enumerate(set(labels))}

acc_kfold = []
f1_kfold = []
recall_kfold = []
precision_kfold = []
data_kfold = []
labels_preds_kfold = []

for idx, (train_idx, test_idx) in enumerate(skf.split(mfccs, labels)):

    mfcc_train, mfcc_test = mfccs[train_idx], mfccs[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    train_labels = [EMO_CLASSES[label] for label in train_labels]
    test_labels = [EMO_CLASSES[label] for label in test_labels]


    train_dataset = EmoDataset(mfcc_train, train_labels)
    test_dataset = EmoDataset(mfcc_test, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Dual()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    acc = []
    f1 = []
    recall = []
    precision = []
    best_labels_preds = []

    for epoch in range(epochs):
        tt_loss = 0
        model.train()
        for mfcc, label in train_loader:

            mfcc = mfcc.unsqueeze(1).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(mfcc)

            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            tt_loss += loss.item()

        print("K-Fold: ", idx, "Epoch: ", epoch, "Train loss: ", tt_loss / len(train_loader))

        model.eval()

        with torch.no_grad():
            all_labels = []
            all_preds = []

            for mfcc, label in test_loader:

                mfcc = mfcc.unsqueeze(1).to(device)
                label = label.to(device)

                output = model(mfcc)
                _, predicted = torch.max(output, 1)

                all_labels.extend(label.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())


            validation_acc = accuracy_score(all_labels, all_preds)
            recall_val = recall_score(all_labels, all_preds, average=avarage)
            f1_val = f1_score(all_labels, all_preds, average=avarage)
            precision_val = precision_score(all_labels, all_preds, average=avarage)

            acc.append(validation_acc)
            f1.append(f1_val)
            recall.append(recall_val)
            precision.append(precision_val)


            if validation_acc > best_acc:
                best_acc = validation_acc
                best_recall = recall_val
                best_f1 = f1_val
                best_precision = precision_val
                best_labels_preds = [all_labels, all_preds]

                # torch.save(model.state_dict(), f"best_model_{idx}.pth")

            print(f"K-Fold: {idx}, Epoch: {epoch}, Accuracy: {validation_acc}, Recall: {recall_val}, F1: {f1_val}, Precision: {precision_val}")

    data_kfold.append([acc, f1, recall, precision])

    acc_kfold.append(best_acc)
    f1_kfold.append(best_f1)
    recall_kfold.append(best_recall)
    precision_kfold.append(best_precision)
    labels_preds_kfold.append(best_labels_preds)

print_classification_report(labels_preds_kfold)
average_classification_report(labels_preds_kfold)
plot_confusion_matrix(labels_preds_kfold)

ua, wa = compute_ua_wa(labels_preds_kfold)
print(f"Unweighted Accuracy: {ua}, Weighted Accuracy: {wa}")