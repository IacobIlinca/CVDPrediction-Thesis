from joblib import load

from prepareDF import X_testHF

# Load the model from the file
pipeline = load('trained_model.joblib')


row_to_predict = X_testHF.iloc[0:1]  # Selects the first row and keeps it in DataFrame format

# Predict using your trained pipeline
prediction = pipeline.predict_proba(row_to_predict)

print(prediction)