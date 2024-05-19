import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, \
    recall_score, precision_score

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF, X_trainHD, X_testHD, y_trainHD, y_testHD, X_trainF, \
    X_testF, y_trainF, y_testF


def applyExtTree(X_train, X_test, y_train, y_test, dt):
    # Combine the train and test data for cross-validation
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    # Initialize the classifier
    classifier = ExtraTreesClassifier()

    # Setting up 10-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(classifier, X, y, cv=kf, scoring='accuracy')
    print(f"10-Fold Cross-Validation Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Fit model on entire dataset for classification report
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Classification report:\n", classification_report(y_test, y_pred))

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


# Apply to different datasets
applyExtTree(X_trainHF, X_testHF, y_trainHF, y_testHF, "HF")
applyExtTree(X_trainHD, X_testHD, y_trainHD, y_testHD, "HD")
applyExtTree(X_trainF, X_testF, y_trainF, y_testF, "F")
