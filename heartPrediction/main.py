from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from models.hyperparameterOptimization import X_trainHF, y_trainHF, X_testHF, y_testHF
best_params= {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_leaf_nodes': 40, 'max_samples': 0.7273103361119226, 'bootstrap': True, 'class_weight': 'balanced'}

rndForest = RandomForestClassifier(**best_params)
rndForest.fit(X_trainHF, y_trainHF)
y_pred = rndForest.predict(X_testHF)
randForestAcc = accuracy_score(y_testHF, y_pred)
print("optimised accuracy: " + randForestAcc.__str__())