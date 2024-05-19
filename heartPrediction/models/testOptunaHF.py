import os

import matplotlib.pyplot as plt
import numpy as np
import optuna as optuna
from joblib import dump
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


best_params = {'n_estimators': 11, 'max_depth': 15, 'min_samples_split': 13, 'min_samples_leaf': 6, 'max_features': 'log2'}


pipeline = make_pipeline(SMOTE(random_state=42), ExtraTreesClassifier(**best_params, random_state=42))
pipeline.fit(X_trainHF, y_trainHF)

dump(pipeline, 'trained_model.joblib')

y_pred = pipeline.predict(X_testHF)

# After obtaining best_params and re-training your model
y_train_pred = pipeline.predict(X_trainHF)
y_test_pred = pipeline.predict(X_testHF)

# Compute F1 scores for both training and test sets
f1_train = f1_score(y_trainHF, y_train_pred, average='weighted')
f1_test = f1_score(y_testHF, y_test_pred, average='weighted')

# Display the results
print(f"Training F1 Score: {f1_train}")
print(f"Test F1 Score: {f1_test}")

# Metrics
accuracy = accuracy_score(y_testHF, y_pred)
f1 = f1_score(y_testHF, y_pred, average='weighted')
recall = recall_score(y_testHF, y_pred, average='weighted')
precision = precision_score(y_testHF, y_pred, average='weighted')
print(f"Accuracy: {accuracy}, F1 Score: {f1}, Recall: {recall}, Precision: {precision}")

# Confusion Matrix
cm = confusion_matrix(y_testHF, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('../results/modelResults/optuna/best/confusion_matrix' + "HF" + '.png')

# Metrics Table
metrics_data = {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall], 'Precision': [precision]}
metrics_df = pd.DataFrame(metrics_data)
print(metrics_df)
# Write metrics to file
review_data = {'Classififer': ['Extra Tree with Optuna'], 'DataSet': ["HF"], 'Accuracy': [accuracy], 'F1 Score': [f1],
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
plt.savefig('../results/modelResults/optuna/best/metrics_table' + "HF" + '.png')
