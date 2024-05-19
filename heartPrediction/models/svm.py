import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, ConfusionMatrixDisplay

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF, X_trainHD, X_testHD, y_trainHD, y_testHD, X_trainF, \
    X_testF, y_trainF, y_testF

def applySVM(X_train, X_test, y_train, y_test, dt):
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize SVM with best known parameters
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    # Apply mutual information feature selection
    selector = SelectKBest(mutual_info_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Now apply the SVM with RBF kernel

    svm.fit(X_train_selected, y_train)
    y_pred = svm.predict(X_test_selected)
    print("Classification report for SVM is \n" + classification_report(y_test, y_pred))

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('../results/modelResults/SVM/confusion_matrix' + dt + '.png')

    # Metrics Table
    metrics_data = {'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall], 'Precision': [precision]}
    metrics_df = pd.DataFrame(metrics_data)

    # Write metrics to file
    review_data = {'Classifier': ['SVM'], 'DataSet': [dt], 'Accuracy': [accuracy], 'F1 Score': [f1], 'Recall': [recall],
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

    plt.title("SVM Classifier Results")
    plt.savefig('../results/modelResults/SVM/metrics_table' + dt + '.png')

    # Print metrics to console
    print(metrics_df)

#applySVM(X_trainHF, X_testHF, y_trainHF, y_testHF, "HF")
#applySVM(X_trainHD, X_testHD, y_trainHD, y_testHD, "HD")
applySVM(X_trainF, X_testF, y_trainF, y_testF, "F")
