#!/usr/bin/env python
# -*-coding:utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd


def calculate_metric(gt, pred):
    pred = pred.copy()
    pred.loc[pred >= 0.5] = 1
    pred.loc[pred < 0.5] = 0
    confusion = confusion_matrix(gt, pred)
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Specificity:', TN / float(TN + FP))


def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['label', 'pred'])
        df.loc[:, 'label'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.label == 1]
        df_neg = df[df.label == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.label.values, neg_sample.label.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics


def print_statistics(label, seq):
    print('-' * 30)
    statistics = bootstrap_auc(label, seq, [0, 1])
    print("均值:", np.mean(statistics, axis=1))
    print("最大值:", np.max(statistics, axis=1))
    print("最小值:", np.min(statistics, axis=1))
    print("\n")


if __name__ == '__main__':
    goal = '2ymace'
    print('*' * 20)
    print("GBDT")
    df = pd.read_csv('res/{}/GBDT_preds.csv'.format(goal))
    prediction = df['prediction']
    label = df["labels"]
    calculate_metric(label, prediction)
    print_statistics(label, prediction)
    print('*' * 20)

    print("AdaBoost")
    df = pd.read_csv('res/{}/AdaBoost_preds.csv'.format(goal))
    prediction = df['prediction']
    label = df["labels"]
    calculate_metric(label, prediction)
    print_statistics(label, prediction)
    print('*' * 20)

    print("SVM")
    df = pd.read_csv('res/{}/SVM_preds.csv'.format(goal))
    prediction = df['prediction']
    label = df["labels"]
    calculate_metric(label, prediction)
    print_statistics(label, prediction)
    print('*' * 20)

    print("RF")
    df = pd.read_csv('res/{}/RF_preds.csv'.format(goal))
    prediction = df['prediction']
    label = df["labels"]
    calculate_metric(label, prediction)
    print_statistics(label, prediction)
    print('*' * 20)

    print("LR")
    df = pd.read_csv('res/{}/LR_preds.csv'.format(goal))
    prediction = df['prediction']
    label = df["labels"]
    calculate_metric(label, prediction)
    print_statistics(label, prediction)
