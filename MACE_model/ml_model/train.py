#!/usr/bin/env python
# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold

import joblib
from sklearn.svm import SVC, SVR
from cal_index import bootstrap_auc, calculate_metric

goal = '2ymace'
data = pd.read_csv('data/result.csv', header=None)
data = data.sample(frac=1)

x = data.iloc[:, :3]
y = data.iloc[:, 3]
X = np.array(x.values)
y = np.array(y.values)

# Model
model_rf = RandomForestRegressor(n_estimators=10)
model_lr = LinearRegression()
# model_kn = KNeighborsClassifier()
model_svc = SVR(kernel='poly',
                )

model_abr = AdaBoostRegressor(learning_rate=0.01, n_estimators=200, random_state=42)
# model_xg = XGBClassifier()
model_gbr = GradientBoostingRegressor(learning_rate=0.01, n_estimators=210, max_depth=3, min_samples_leaf=30
                                      , min_samples_split=70, subsample=0.7
                                      , random_state=42)

cv = StratifiedKFold(n_splits=10)


def train(model_name, classifier):
    preds = []
    labels = []
    classifier = classifier
    for i, (train, test) in enumerate(cv.split(X, y)):
        cly = classifier.fit(X[train], y[train])
        joblib.dump(cly, 'model/{}/{}.pkl'.format(goal, model_name))

        model = joblib.load('model/{}/{}.pkl'.format(goal, model_name))
        pred = model.predict(X[test])
        preds = preds + list(pred)
        labels = labels + list(y[test])
    data = {
        'prediction': preds,
        'labels': labels
    }
    df = pd.DataFrame(data)
    df.to_csv('res/{}/{}_preds.csv'.format(goal, model_name))


models = {
    'SVM': model_svc,
    'GBDT': model_gbr,
    'LR': model_lr,
    'RF': model_rf,
    'AdaBoost': model_abr
}


def make_auc(df):
    prediction = df['prediction']
    label = df["labels"]
    statistics = bootstrap_auc(label, prediction, [0, 1])
    aucs = np.mean(statistics, axis=1)
    max_aucs = np.max(statistics, axis=1)
    min_aucs = np.min(statistics, axis=1)
    auc_1 = aucs[0]
    max_auc = max_aucs[0]
    min_auc = min_aucs[0]

    if aucs[0] < aucs[1]:
        auc_1 = aucs[1]
        max_auc = max_aucs[1]
        min_auc = min_aucs[1]
    fpr, tpr, thersholds = roc_curve(label, prediction, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('------')
    print('fpr:', 1 - np.mean(fpr))
    print('tpr:', np.mean(tpr))
    print('auc: ', roc_auc)
    # print('auc2:',auc_1)
    # print('max_auc: ',max_auc)
    # print('min_auc: ',min_auc)
    # calculate_metric(label,prediction)
    return roc_auc, min_auc, max_auc, fpr, tpr


if __name__ == '__main__':
    train('GBDT', models['GBDT'])
    train('AdaBoost', models['AdaBoost'])
    train('SVM', models['SVM'])
    train('RF', models['RF'])
    train('LR', models['LR'])

    path = 'res/{}/'.format(goal)

    # Machine learning Models
    df1 = pd.read_csv(path + 'GBDT_preds.csv')
    df2 = pd.read_csv(path + 'AdaBoost_preds.csv')
    df3 = pd.read_csv(path + 'SVM_preds.csv')
    df4 = pd.read_csv(path + 'RF_preds.csv')
    df5 = pd.read_csv(path + 'LR_preds.csv')
    roc_gbdt, min_1, max_1, fpr_1, tpr_1 = make_auc(df1)
    roc_AdaBoost, min_2, max_2, fpr_2, tpr_2 = make_auc(df2)
    roc_svm, min_3, max_3, fpr_3, tpr_3 = make_auc(df3)
    roc_rf, min_4, max_4, fpr_4, tpr_4 = make_auc(df4)
    roc_lr, min_5, max_5, fpr_5, tpr_5 = make_auc(df5)
