import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataAnalysys.heartFailureDS import df_new as dfHeartFailure
from dataAnalysys.heartDiseaseDS import df_new as dfHeartDisease
from dataAnalysys.FraminghamDS import df_new as dfFramingham


def remove_outliers(df, feature_columns, contamination_fraction=0.05):
    # Isolation Forest for outlier detection
    iso_forest = IsolationForest(contamination=contamination_fraction, random_state=42)

    # Fit only on the feature columns to keep the method general
    iso_forest.fit(df[feature_columns])

    # Predictions: -1 for outliers, 1 for inliers
    preds = iso_forest.predict(df[feature_columns])

    # Remove outliers
    df_filtered = df[preds == 1]
    return df_filtered

def scale_df(df, numcols):
    scaler = StandardScaler()
    df[numcols] = scaler.fit_transform(df[numcols])


def splitTestAndTrain(df, targetColumn):
    # Set up X and y variables
    y, X = df[targetColumn], df.drop(columns=targetColumn, axis=1)

    # Split the data into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def fill_empty_with_mean(df, columns):
    # Replace empty strings with NaN
    df.replace('', pd.NA, inplace=True)

    # Convert columns to numeric (in case they are not already)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Calculate the mean for each column
    mean_values = df.mean()

    # Replace empty values with the mean average
    df.fillna(mean_values, inplace=True)

    return df

numcolsHeartFailure = ['RestingBP', 'Cholesterol', 'MaxHR']
targetColumnHeartFailure = 'HeartDisease'
new_dfHeartFailure = fill_empty_with_mean(dfHeartFailure, numcolsHeartFailure)
new_dfHeartFailure_no_outliers = remove_outliers(new_dfHeartFailure, numcolsHeartFailure)
scale_df(new_dfHeartFailure_no_outliers, numcolsHeartFailure)
X_trainHF, X_testHF, y_trainHF, y_testHF = splitTestAndTrain(new_dfHeartFailure_no_outliers, targetColumnHeartFailure)


numcolsHeartDisease = ['trestbps', 'chol', 'thalach', 'oldpeak']
targetColumnHeartDisease = 'target'
new_dfHeartDisease = fill_empty_with_mean(dfHeartDisease, numcolsHeartDisease)
new_dfHeartDisease_no_outliers = remove_outliers(dfHeartDisease, numcolsHeartDisease)
scale_df(dfHeartDisease, numcolsHeartDisease)
X_trainHD, X_testHD, y_trainHD, y_testHD = splitTestAndTrain(dfHeartDisease, targetColumnHeartDisease)

numcolsFramingham = ['ht', 'wt', 'dbp', 'sbp', 'mrw', 'smoke', 'sc1', 'sc2']
targetColumnFramingham = 'cexam'
new_dfFramingham = fill_empty_with_mean(dfFramingham, numcolsFramingham)
new_dfFramingham_no_outliers = remove_outliers(new_dfFramingham, numcolsFramingham)
scale_df(new_dfFramingham_no_outliers, numcolsFramingham)
X_trainF, X_testF, y_trainF, y_testF = splitTestAndTrain(new_dfFramingham_no_outliers, targetColumnFramingham)
