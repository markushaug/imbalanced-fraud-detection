from model_utils.evaluation import get_metrics, evaluate_model

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, fbeta_score, matthews_corrcoef, accuracy_score, roc_curve, precision_recall_curve, auc, classification_report, average_precision_score
from imblearn.metrics import geometric_mean_score
import pandas as pd
from imbens.ensemble import SelfPacedEnsembleClassifier
from scikeras.wrappers import KerasClassifier

import keras
from xgboost import XGBClassifier
from keras import Sequential

# set seed
SEED=123
np.random.seed(SEED)

# create test set
X, y = make_classification(random_state=SEED, n_classes=2, n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

def assert_dicts_almost_equal(dict1, dict2, precision=0.3):
    assert dict1.keys() == dict2.keys(), "Keys do not match."
    for key in dict1:
        if isinstance(dict1[key], float):
            assert abs(dict1[key] - dict2[key]) < precision, f"Mismatch at {key}: {dict1[key]} vs {dict2[key]}"
        else:
            assert dict1[key] == dict2[key], f"Mismatch at {key}: {dict1[key]} vs {dict2[key]}"

# XGB Classifier
def test_get_metrics_with_xgb_classifier():
	# given
	vanilla_xgb = XGBClassifier(random_state=SEED)
	vanilla_xgb.fit(X_train, y_train)

	xgb_scores = vanilla_xgb.predict_proba(X_test)
	xgb_preds = vanilla_xgb.predict(X_test)
    
	expected_xgb_metrics = {}
	y_scores = np.array(xgb_scores[:, 1]).reshape(-1,1)

	cm = confusion_matrix(y_test, xgb_preds)
	tn, fp, fn, tp = cm.ravel().astype('float32')

	# add to dic
	expected_xgb_metrics["tn"] = tn
	expected_xgb_metrics["fp"] = fp
	expected_xgb_metrics["fn"] = fn
	expected_xgb_metrics["tp"] = tp

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	expected_xgb_metrics["precision"] = round(precision, 4)
	expected_xgb_metrics["recall"] = round(recall, 4)

	precision_array, recall_array, _ = precision_recall_curve(y_test, y_scores)
	expected_xgb_metrics["AUCPRC"] = round(auc(recall_array, precision_array), 4)

	expected_xgb_metrics["F1"] = round(fbeta_score(y_test, xgb_preds, beta=1), 4)
	expected_xgb_metrics["ROCAUC"] = round(roc_auc_score(y_test, y_scores), 4)
	expected_xgb_metrics["MCC"] = round(matthews_corrcoef(y_test, xgb_preds), 4)
	expected_xgb_metrics["ACC"] = round(accuracy_score(y_test, xgb_preds), 4)
	expected_xgb_metrics["GMEAN"] = round(geometric_mean_score(y_test, xgb_preds), 4)
	
	# when
	actual_xgb_metrics = get_metrics(y_test, xgb_scores[:, 1])
        
	# then
	assert expected_xgb_metrics == actual_xgb_metrics

def test_evaluate_model_with_xgb_classifier():
	# given
	vanilla_xgb = XGBClassifier(random_state=SEED)
	vanilla_xgb.fit(X_train, y_train)

	xgb_scores = vanilla_xgb.predict_proba(X_test)
	xgb_preds = vanilla_xgb.predict(X_test)

	expected_xgb_metrics = {}
	y_scores = np.array(xgb_scores[:, 1]).reshape(-1,1)

	cm = confusion_matrix(y_test, xgb_preds)
	tn, fp, fn, tp = cm.ravel().astype('float32')

	# add to dic
	expected_xgb_metrics["tn"] = tn
	expected_xgb_metrics["fp"] = fp
	expected_xgb_metrics["fn"] = fn
	expected_xgb_metrics["tp"] = tp

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	expected_xgb_metrics["precision"] = round(precision, 4)
	expected_xgb_metrics["recall"] = round(recall, 4)

	precision_array, recall_array, _ = precision_recall_curve(y_test, y_scores)
	expected_xgb_metrics["AUCPRC"] = round(auc(recall_array, precision_array), 4)

	expected_xgb_metrics["F1"] = round(fbeta_score(y_test, xgb_preds, beta=1), 4)
	expected_xgb_metrics["ROCAUC"] = round(roc_auc_score(y_test, y_scores), 4)
	expected_xgb_metrics["MCC"] = round(matthews_corrcoef(y_test, xgb_preds), 4)
	expected_xgb_metrics["ACC"] = round(accuracy_score(y_test, xgb_preds), 4)
	expected_xgb_metrics["GMEAN"] = round(geometric_mean_score(y_test, xgb_preds), 4)
	expected_xgb_metrics['model_name'] = "XGBClassifier"
		
	# when
	actual_xgb_metrics = evaluate_model(vanilla_xgb, X_test, y_test)
		
	# then
	assert expected_xgb_metrics == actual_xgb_metrics[0]  

# Keras Sequential
def test_get_metrics_with_keras_mlp():
    # given
	mlp = keras.Sequential(
		[
			keras.Input(shape=(X.shape[1],)),
			keras.layers.Dense(128, activation="relu"),
			keras.layers.Dense(64, activation="relu"),
			keras.layers.Dense(1, activation="sigmoid"),
		]
	)

	metrics = [
		keras.metrics.AUC(name='auc_prc', curve="PR"),
	]

	mlp.compile(
		optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=metrics
	)

	hist = mlp.fit(X_train, y_train, epochs=20, verbose=0)

	mlp_scores = mlp.predict(X_test)
	mlp_preds = (mlp_scores > 0.5).astype(int)

	# # ----------
	# # calc scores
	# # -----------

	expected_mlp_metrics = {}

	cm = confusion_matrix(y_test, mlp_preds.flatten())
	tn, fp, fn, tp = cm.ravel().astype('float32')

	# add to dic
	expected_mlp_metrics["tn"] = tn
	expected_mlp_metrics["fp"] = fp
	expected_mlp_metrics["fn"] = fn
	expected_mlp_metrics["tp"] = tp

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	expected_mlp_metrics["precision"] = round(precision, 4)
	expected_mlp_metrics["recall"] = round(recall, 4)

	precision_array, recall_array, _ = precision_recall_curve(y_test, mlp_scores)
	expected_mlp_metrics["AUCPRC"] = round(auc(recall_array, precision_array), 4)

	expected_mlp_metrics["F1"] = round(fbeta_score(y_test, mlp_preds, beta=1), 4)
	expected_mlp_metrics["ROCAUC"] = round(roc_auc_score(y_test, mlp_scores), 4)
	expected_mlp_metrics["MCC"] = round(matthews_corrcoef(y_test, mlp_preds), 4)
	expected_mlp_metrics["ACC"] = round(accuracy_score(y_test, mlp_preds), 4)
	expected_mlp_metrics["GMEAN"] = round(geometric_mean_score(y_test, mlp_preds), 4)
      
	# when 
	actual_mlp_metrics = get_metrics(y_test, mlp_scores.flatten(), threshold=0.5, op=">")

	# then
	assert_dicts_almost_equal(expected_mlp_metrics, actual_mlp_metrics, precision=0.3)	
      
def test_evaluate_model_with_keras_mlp():
    # given
	mlp = keras.Sequential(
		[
			keras.Input(shape=(X.shape[1],)),
			keras.layers.Dense(128, activation="relu"),
			keras.layers.Dense(64, activation="relu"),
			keras.layers.Dense(1, activation="sigmoid"),
		]
	)

	metrics = [
		keras.metrics.AUC(name='auc_prc', curve="PR"),
	]

	mlp.compile(
		optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=metrics
	)

	hist = mlp.fit(X_train, y_train, epochs=20, verbose=0)

	mlp_scores = mlp.predict(X_test)
	mlp_preds = (mlp_scores > 0.5).astype(int)

	# # ----------
	# # calc scores
	# # -----------

	expected_mlp_metrics = {}

	cm = confusion_matrix(y_test, mlp_preds.flatten())
	tn, fp, fn, tp = cm.ravel().astype('float32')

	# add to dic
	expected_mlp_metrics["tn"] = tn
	expected_mlp_metrics["fp"] = fp
	expected_mlp_metrics["fn"] = fn
	expected_mlp_metrics["tp"] = tp

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	expected_mlp_metrics["precision"] = round(precision, 4)
	expected_mlp_metrics["recall"] = round(recall, 4)

	precision_array, recall_array, _ = precision_recall_curve(y_test, mlp_scores)
	expected_mlp_metrics["AUCPRC"] = round(auc(recall_array, precision_array), 4)

	expected_mlp_metrics["F1"] = round(fbeta_score(y_test, mlp_preds, beta=1), 4)
	expected_mlp_metrics["ROCAUC"] = round(roc_auc_score(y_test, mlp_scores), 4)
	expected_mlp_metrics["MCC"] = round(matthews_corrcoef(y_test, mlp_preds), 4)
	expected_mlp_metrics["ACC"] = round(accuracy_score(y_test, mlp_preds), 4)
	expected_mlp_metrics["GMEAN"] = round(geometric_mean_score(y_test, mlp_preds), 4)
      
	# when 
	expected_mlp_metrics['model_name'] = "Sequential"
	actual_mlp_metrics = evaluate_model(mlp, X_test, y_test, threshold=0.5, op=">")

	# then
	assert_dicts_almost_equal(expected_mlp_metrics, actual_mlp_metrics[0], precision=0.3)	

# KerasWrapper: KerasClassifier
def test_get_metrics_with_keras_clf():
    # given
	mlp = keras.Sequential(
		[
			keras.Input(shape=(X_train.shape[1],)),
			keras.layers.Dense(128, activation="relu"),
			keras.layers.Dense(64, activation="relu"),
			keras.layers.Dense(1, activation="sigmoid"),
		]
	)

	metrics = [
		keras.metrics.AUC(name='auc_prc', curve="PR"),
	]

	mlp.compile(
		optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=metrics
	)

	hist = mlp.fit(X_train, y_train, epochs=20, verbose=0)

	mlp_clf = KerasClassifier(mlp,
					  epochs=20, 
					  optimizer=keras.optimizers.Adam(), 
					  metrics=metrics,
					  loss="binary_crossentropy",
					  random_state=SEED
					  )
	
	# Initialize Instance. Note: Do not run mlp.fit(). This would retrain the model.
	mlp_clf.initialize(X_train, y_train)

	mlp_clf_scores = mlp_clf.predict_proba(X_test)[:, 1]
	mlp_clf_preds = mlp_clf.predict(X_test)

	# # ----------
	# # calc scores
	# # -----------

	expected_mlp_clf_metrics = {}

	cm = confusion_matrix(y_test, mlp_clf_preds.flatten())
	tn, fp, fn, tp = cm.ravel().astype('float32')

	# add to dic
	expected_mlp_clf_metrics["tn"] = tn
	expected_mlp_clf_metrics["fp"] = fp
	expected_mlp_clf_metrics["fn"] = fn
	expected_mlp_clf_metrics["tp"] = tp

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	expected_mlp_clf_metrics["precision"] = round(precision, 4)
	expected_mlp_clf_metrics["recall"] = round(recall, 4)

	precision_array, recall_array, _ = precision_recall_curve(y_test, mlp_clf_preds)
	expected_mlp_clf_metrics["AUCPRC"] = round(auc(recall_array, precision_array), 4)

	expected_mlp_clf_metrics["F1"] = round(fbeta_score(y_test, mlp_clf_preds, beta=1), 4)
	expected_mlp_clf_metrics["ROCAUC"] = round(roc_auc_score(y_test, mlp_clf_scores), 4)
	expected_mlp_clf_metrics["MCC"] = round(matthews_corrcoef(y_test, mlp_clf_preds), 4)
	expected_mlp_clf_metrics["ACC"] = round(accuracy_score(y_test, mlp_clf_preds), 4)
	expected_mlp_clf_metrics["GMEAN"] = round(geometric_mean_score(y_test, mlp_clf_preds), 4)
      
	# when 
	actual_mlp_clf_metrics = get_metrics(y_test, mlp_clf_scores, threshold=0.5, op=">")

	# then
	
    # Due to floating point precision, we have to allow a tolerance here of 0.3
	assert_dicts_almost_equal(expected_mlp_clf_metrics, actual_mlp_clf_metrics, precision=0.3)
      
def test_evaluate_model_with_keras_clf():     
    # given
	mlp = keras.Sequential(
		[
			keras.Input(shape=(X_train.shape[1],)),
			keras.layers.Dense(128, activation="relu"),
			keras.layers.Dense(64, activation="relu"),
			keras.layers.Dense(1, activation="sigmoid"),
		]
	)

	metrics = [
		keras.metrics.AUC(name='auc_prc', curve="PR"),
	]

	mlp.compile(
		optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=metrics
	)

	hist = mlp.fit(X_train, y_train, epochs=20, verbose=0)

	mlp_clf = KerasClassifier(mlp,
					  epochs=20, 
					  optimizer=keras.optimizers.Adam(), 
					  metrics=metrics,
					  loss="binary_crossentropy",
					  random_state=SEED
					  )
	
	# Initialize Instance. Note: Do not run mlp.fit(). This would retrain the model.
	mlp_clf.initialize(X_train, y_train)

	mlp_clf_scores = mlp_clf.predict_proba(X_test)[:,1]
	mlp_clf_preds = mlp_clf.predict(X_test)

	# # ----------
	# # calc scores
	# # -----------

	expected_mlp_clf_metrics = {}

	cm = confusion_matrix(y_test, mlp_clf_preds.flatten())
	tn, fp, fn, tp = cm.ravel().astype('float32')

	# add to dic
	expected_mlp_clf_metrics["tn"] = tn
	expected_mlp_clf_metrics["fp"] = fp
	expected_mlp_clf_metrics["fn"] = fn
	expected_mlp_clf_metrics["tp"] = tp

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	expected_mlp_clf_metrics["precision"] = round(precision, 4)
	expected_mlp_clf_metrics["recall"] = round(recall, 4)

	precision_array, recall_array, _ = precision_recall_curve(y_test, mlp_clf_preds)
	expected_mlp_clf_metrics["AUCPRC"] = round(auc(recall_array, precision_array), 4)

	expected_mlp_clf_metrics["F1"] = round(fbeta_score(y_test, mlp_clf_preds, beta=1), 4)
	expected_mlp_clf_metrics["ROCAUC"] = round(roc_auc_score(y_test, mlp_clf_scores), 4)
	expected_mlp_clf_metrics["MCC"] = round(matthews_corrcoef(y_test, mlp_clf_preds), 4)
	expected_mlp_clf_metrics["ACC"] = round(accuracy_score(y_test, mlp_clf_preds), 4)
	expected_mlp_clf_metrics["GMEAN"] = round(geometric_mean_score(y_test, mlp_clf_preds), 4)
      
	# when 
	actual_mlp_clf_metrics = evaluate_model(mlp_clf, X_test, y_test, threshold=0.5, op=">")
	expected_mlp_clf_metrics['model_name'] = 'KerasClassifier'
	
	# then
	
    # Due to floating point precision, we have to allow a tolerance here of 0.3
	assert_dicts_almost_equal(expected_mlp_clf_metrics, actual_mlp_clf_metrics[0], precision=0.3)
      