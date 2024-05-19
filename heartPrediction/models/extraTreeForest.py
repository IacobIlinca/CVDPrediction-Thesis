import os

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF, X_trainHD, X_testHD, y_trainHD, y_testHD, X_trainF, \
    X_testF, y_trainF, y_testF


def applyExtraTreeForest(X_train, X_test, y_train, y_test, dt):
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40],
    }

    smote = SMOTE()
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    extra_tree_forest = ExtraTreesClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=extra_tree_forest, param_grid=param_grid, cv=5, scoring='f1',
                               n_jobs=-1)
    grid_search.fit(X_train_bal, y_train_bal)

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)


    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Write metrics to file
    review_data = {'Classififer': ['Extra Tree'], 'DataSet': [dt], 'Accuracy': [accuracy], 'F1 Score': [f1],
                   'Recall': [recall],
                   'Precision': [precision]}

    df = pd.DataFrame(review_data)
    file_path = '../results/classifier_metrics.csv'
    append = True
    if append and os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', index=False)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('../results/modelResults/extraTree/confusion_matrix' + dt + '.png')

    # Metrics Table
    metrics_data = {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall], 'Precision': [precision]}
    metrics_df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(7, 3))  # Adjust size to fit your needs
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values,
             colLabels=metrics_df.columns,
             loc='center')

    plt.title("Extra Tree Classifier Results")
    plt.savefig('../results/modelResults/extraTree/metrics_table' + dt + '.png')

    # Print metrics to console
    print(metrics_df)


applyExtraTreeForest(X_trainHF, X_testHF, y_trainHF, y_testHF, "HF")

applyExtraTreeForest(X_trainHD, X_testHD, y_trainHD, y_testHD, "HD")

applyExtraTreeForest(X_trainF, X_testF, y_trainF, y_testF, "F")
