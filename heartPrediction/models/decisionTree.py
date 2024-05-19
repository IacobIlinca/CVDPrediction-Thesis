import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF, X_trainHD, X_testHD, y_trainHD, y_testHD, X_trainF, \
    X_testF, y_trainF, y_testF


def applyDecisionTreeWithKFold(X_train, X_test, y_train, y_test, dt):
    # Define the base model
    decisionTree = DecisionTreeClassifier(random_state=42)

    # Define the hyperparameters and their ranges to test
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    # Setup GridSearchCV with k-fold cross-validation, k=5 here as an example
    grid_search = GridSearchCV(estimator=decisionTree, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Find the best hyperparameters
    print("Best parameters:", grid_search.best_params_)

    # Use the best model to make predictions on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate and print the accuracy
    decisionTreeAcc = accuracy_score(y_test, y_pred)
    print("Accuracy of Decision Tree with best hyperparameters:", decisionTreeAcc)

    # Print the classification report
    print("Classification report for Decision Tree:\n", classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('../results/modelResults/decisionTree/confusion_matrix' + dt + '.png')

    # Metrics Table
    metrics_data = {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall], 'Precision': [precision]}
    metrics_df = pd.DataFrame(metrics_data)

    # Write metrics to file
    review_data = {'Classififer': ['Decision Tree'], 'DataSet': [dt], 'Accuracy': [accuracy], 'F1 Score': [f1],
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

    plt.title("Decision Tree Classifier Results")
    plt.savefig('../results/modelResults/decisionTree/metrics_table' + dt + '.png')

    # Print metrics to console
    print(metrics_df)


print("Decision Tree accuracy for Heart Failure dataset:")
applyDecisionTreeWithKFold(X_trainHF, X_testHF, y_trainHF, y_testHF, "HF")

print("Decision Tree accuracy for Heart Disease dataset:")
applyDecisionTreeWithKFold(X_trainHD, X_testHD, y_trainHD, y_testHD, "HD")

print("Decision Tree accuracy for Framingham dataset:")
applyDecisionTreeWithKFold(X_trainF, X_testF, y_trainF, y_testF, "F")
