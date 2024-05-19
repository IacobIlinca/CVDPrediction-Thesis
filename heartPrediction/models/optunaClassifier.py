import os

import matplotlib.pyplot as plt
import numpy as np
import optuna as optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import StratifiedKFold

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF, X_trainHD, X_testHD, y_trainHD, y_testHD, X_trainF, \
    X_testF, y_trainF, y_testF


# def objective(trial, X, y, n_splits=5):
#     """
#     Objective function for Optuna hyperparameter search.
#     """
#     # Hyperparameters to optimize
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 10, 300),
#         'max_depth': trial.suggest_int('max_depth', 10, 40, step=10) or None,
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
#     }
def objective(trial, X, y, n_splits=10):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),  # Reduced upper limit
        'max_depth': trial.suggest_int('max_depth', 5, 20, step=5),  # Lower depths
        'min_samples_split': trial.suggest_int('min_samples_split', 4, 20),  # Higher values
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),  # Higher values
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }

    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1_scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # SMOTE for balancing the training set
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        model = ExtraTreesClassifier(**params, random_state=42)
        model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_valid)
        f1 = f1_score(y_valid, y_pred, average='weighted')
        f1_scores.append(f1)

    return np.mean(f1_scores)


def applyExtraTreeForest(X_train, X_test, y_train, y_test, dt):
    # Optuna optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose output
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100, show_progress_bar=True)

    print("Best parameters:", study.best_params)
    print("Best F1 score:", study.best_value)

    # Train model with best parameters
    best_params = study.best_params
    print("params importance: ")
    print(optuna.importance.get_param_importances(study))
    pipeline = make_pipeline(SMOTE(random_state=42), ExtraTreesClassifier(**best_params, random_state=42))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # After obtaining best_params and re-training your model
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Compute F1 scores for both training and test sets
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')

    # Display the results
    print(f"Training F1 Score: {f1_train}")
    print(f"Test F1 Score: {f1_test}")

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy}, F1 Score: {f1}, Recall: {recall}, Precision: {precision}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('../results/modelResults/optuna/confusion_matrix' + dt + '.png')

    # Metrics Table
    metrics_data = {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall], 'Precision': [precision]}
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df)
    # Write metrics to file
    review_data = {'Classififer': ['Extra Tree with Optuna'], 'DataSet': [dt], 'Accuracy': [accuracy], 'F1 Score': [f1],
                   'Recall': [recall],
                   'Precision': [precision]}

    df = pd.DataFrame(review_data)
    file_path = '../results/classifier_metrics.csv'
    append = True
    if append and os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', index=False)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
    plt.title("Extra Tree Classifier Results")
    plt.savefig('../results/modelResults/optuna/metrics_table' + dt + '.png')
    return metrics_data


applyExtraTreeForest(X_trainF, X_testF, y_trainF, y_testF, "F")

#applyExtraTreeForest(X_trainHF, X_testHF, y_trainHF, y_testHF, "HF")

#applyExtraTreeForest(X_trainHD, X_testHD, y_trainHD, y_testHD, "HD")
