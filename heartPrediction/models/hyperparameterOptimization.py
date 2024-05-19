import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from prepareDF import X_trainHF, X_testHF, y_trainHF, y_testHF
from prepareDF import X_trainHD, X_testHD, y_trainHD, y_testHD
from prepareDF import X_trainF, X_testF, y_trainF, y_testF


def objective(params):
    # Ensure integer parameters are cast to int because hp.quniform includes a float
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    model = RandomForestClassifier(**params)
    model.fit(X_trainHD, y_trainHD)
    score = model.score(X_testHD, y_testHD)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

trials = Trials()
best = fmin(fn=objective,
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
            max_evals=300,
            trials=trials)

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
    'min_samples_leaf': int(best['min_samples_leaf']),    # Ensure integer
    'max_features': max_features_choices[best['max_features']],
    'max_leaf_nodes': max_leaf_nodes_choices[best['max_leaf_nodes']],
    'max_samples': best['max_samples'],  # This was using hp.uniform, so it's directly from 'best'
    'bootstrap': bootstrap_choices[best['bootstrap']],
    'class_weight': class_weight_choices[best['class_weight']],
}

print("Best Parameters:", best_params)

#best_params= {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_leaf_nodes': 40, 'max_samples': 0.7273103361119226, 'bootstrap': True, 'class_weight': 'balanced'}

rndForest = RandomForestClassifier(**best_params)
rndForest.fit(X_trainHD, y_trainHD)
y_pred = rndForest.predict(X_testHD)
randForestAcc = accuracy_score(y_testHD, y_pred)
print("optimised accuracy: " + randForestAcc.__str__())
