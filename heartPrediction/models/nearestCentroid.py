import os

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.neighbors import NearestCentroid

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF, X_trainHD, X_testHD, y_trainHD, y_testHD, X_trainF, \
    X_testF, y_trainF, y_testF


def applyNearestCentroid(X_train, X_test, y_train, y_test, dt):
    smote = SMOTE()
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    nearCen = NearestCentroid()
    nearCen.fit(X_train_bal, y_train_bal)

    y_pred = nearCen.predict(X_test)
    logregAcc = accuracy_score(y_test, y_pred)
    print(logregAcc)
    print("classification_report for logistic regression is \n" + classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('../results/modelResults/nearestCentroid/confusion_matrix' + dt + '.png')

    # Metrics Table
    metrics_data = {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall], 'Precision': [precision]}
    metrics_df = pd.DataFrame(metrics_data)

    # Write metrics to file
    review_data = {'Classififer': ['Nearest Centroid'], 'DataSet': [dt], 'Accuracy': [accuracy], 'F1 Score': [f1],
                   'Recall': [recall],
                   'Precision': [precision]}

    df = pd.DataFrame(review_data)
    file_path = '../results/classifier_metrics.csv'
    append = True
    if append and os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', index=False)

    fig, ax = plt.subplots(figsize=(7, 3))  # Adjust size to fit your needs
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values,
             colLabels=metrics_df.columns,
             loc='center')

    plt.title("Extra Tree Classifier Results")
    plt.savefig('../results/modelResults/nearestCentroid/metrics_table' + dt + '.png')

    # Print metrics to console
    print(metrics_df)


applyNearestCentroid(X_trainHF, X_testHF, y_trainHF, y_testHF, "HF")

applyNearestCentroid(X_trainHD, X_testHD, y_trainHD, y_testHD, "HD")

applyNearestCentroid(X_trainF, X_testF, y_trainF, y_testF, "F")
