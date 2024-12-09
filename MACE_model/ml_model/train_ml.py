from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR


def train(model_name, classifier, cv):
    auc_scores = []
    all_preds = []
    all_labels = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        classifier.fit(X_train, y_train)
        joblib.dump(classifier, f'model/{goal}/{model_name}_fold{fold}.pkl')

        model = joblib.load('model/{}/{}.pkl'.format(goal, model_name))
        pred_prob = model.predict(X_test)
        auc = roc_auc_score(y_test, pred_prob)
        auc_scores.append(auc)

        print(f"AUC for {model_name} Fold {fold}: {auc}")

        all_preds.extend(pred_prob)
        all_labels.extend(y_test)

    # Calculate mean and standard deviation of AUC scores
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    print(f"Mean AUC for {model_name}: {mean_auc:.4f}")
    print(f"Standard Deviation of AUC for {model_name}: {std_auc:.4f}")

    data = {
        'prediction_probability': all_preds,
        'labels': all_labels
    }
    df = pd.DataFrame(data)
    df.to_csv(f'res/{goal}/{model_name}_preds.csv', index=False)

    return mean_auc, std_auc


if __name__ == '__main__':
    goal = '2ymace'
    data = pd.read_csv('data/result.csv', header=None)
    data = data.sample(frac=1)
    x = data.iloc[:, :3]
    y = data.iloc[:, 3]
    X = np.array(x.values)
    y = np.array(y.values)
    # Assuming models are defined as before and cv is a StratifiedKFold object with n_splits=10
    cv = StratifiedKFold(n_splits=10)
    model_rf = RandomForestRegressor(n_estimators=10)
    model_lr = LinearRegression()
    model_svc = SVR(kernel='poly',
                    )

    model_abr = AdaBoostRegressor(learning_rate=0.01, n_estimators=200, random_state=42)
    model_gbr = GradientBoostingRegressor(learning_rate=0.01, n_estimators=210, max_depth=3, min_samples_leaf=30
                                          , min_samples_split=70, subsample=0.7
                                          , random_state=42)
    models = {
        'SVM': model_svc,
        'GBDT': model_gbr,
        'LR': model_lr,
        'RF': model_rf,
        'AdaBoost': model_abr
    }

    # Train each model and calculate AUC for each fold, then compute mean and standard deviation
    results = {}
    for name, clf in models.items():
        print("Model: ", name)
        mean_auc, std_auc = train(name, clf, cv)
        results[name] = {'mean_auc': mean_auc, 'std_auc': std_auc}
    print(results)
