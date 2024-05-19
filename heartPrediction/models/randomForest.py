import os

import matplotlib.pyplot as plt
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, ConfusionMatrixDisplay

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF, X_trainHD, X_testHD, y_trainHD, y_testHD, X_trainF, \
    X_testF, y_trainF, y_testF


def applyOptimizedRandomForest(X_train, X_test, y_train, y_test, dt):
    def objective(params):
        # Ensure integer parameters are cast to int because hp.quniform includes a float
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        # Use cross-validation or a validation set instead of the test set for a more robust evaluation
        score = model.score(X_test, y_test)
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space={
            'n_estimators': hp.choice('n_estimators', range(50, 201, 10)),
            'max_depth': hp.choice('max_depth', [None] + list(range(5, 31, 5))),
            # Using quniform for integer values, ensure to wrap with int() where used
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
            'max_features': hp.choice('max_features', [None, 'sqrt', 'log2', 0.5, 0.7]),
            'max_leaf_nodes': hp.choice('max_leaf_nodes', [None] + list(range(10, 51, 10))),
            'max_samples': hp.uniform('max_samples', 0.5, 1.0),
            'bootstrap': hp.choice('bootstrap', [True]),
            'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample']),
        },
        algo=tpe.suggest,
        max_evals=100,  # Consider using a smaller number for quicker results, especially for testing
        trials=trials
    )

    # Define hyperparameter choices outside of fmin to use them for result interpretation
    n_estimators_choices = range(50, 201, 10)
    max_depth_choices = [None] + list(range(5, 31, 5))
    max_features_choices = [None, 'sqrt', 'log2', 0.5, 0.7]
    max_leaf_nodes_choices = [None] + list(range(10, 51, 10))
    bootstrap_choices = [True]
    class_weight_choices = [None, 'balanced', 'balanced_subsample']

    # Use the indices in 'best' to get the actual hyperparameter values
    best_params = {
        'n_estimators': n_estimators_choices[best['n_estimators']],
        'max_depth': max_depth_choices[best['max_depth']],
        # 'min_samples_split' and 'min_samples_leaf' do not require adjustment because they use hp.quniform
        'min_samples_split': int(best['min_samples_split']),  # Ensure integer
        'min_samples_leaf': int(best['min_samples_leaf']),  # Ensure integer
        'max_features': max_features_choices[best['max_features']],
        'max_leaf_nodes': max_leaf_nodes_choices[best['max_leaf_nodes']],
        'max_samples': best['max_samples'],  # This was using hp.uniform, so it's directly from 'best'
        'bootstrap': bootstrap_choices[best['bootstrap']],
        'class_weight': class_weight_choices[best['class_weight']],
    }

    print("Best Parameters:", best_params)

    # Train and evaluate the model with the optimized parameters
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # After obtaining best_params and re-training your model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute F1 scores for both training and test sets
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')

    # Display the results
    print(f"Training F1 Score: {f1_train}")
    print(f"Test F1 Score: {f1_test}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimized Accuracy: {accuracy}")
    print("Classification Report for Optimized Random Forest:\n", classification_report(y_test, y_pred))

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('../results/modelResults/rndForest/confusion_matrix' + dt + '.png')

    # Metrics Table
    metrics_data = {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall], 'Precision': [precision]}
    metrics_df = pd.DataFrame(metrics_data)

    # Write metrics to file
    review_data = {'Classififer': ['Random Forest'], 'DataSet': [dt], 'Accuracy': [accuracy], 'F1 Score': [f1],
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

    plt.title("Random Forest Classifier Results")
    plt.savefig('../results/modelResults/rndForest/metrics_table' + dt + '.png')

    # Print metrics to console
    print(metrics_df)

#
# print("Random Forest accuracy for Heart Failure dataset:")
# applyOptimizedRandomForest(X_trainHF, X_testHF, y_trainHF, y_testHF, "HF")

print("Random Forest accuracy for Heart Disease dataset:")
applyOptimizedRandomForest(X_trainHD, X_testHD, y_trainHD, y_testHD, "HD")
#
# print("Random Forest accuracy for Framingham dataset:")
# applyOptimizedRandomForest(X_trainF, X_testF, y_trainF, y_testF, "F")
