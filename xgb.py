from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,\
    multilabel_confusion_matrix, roc_curve, auc, RocCurveDisplay, DetCurveDisplay
from sklearn import svm, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import xgboost as xgb
Data = pd.read_csv("First_processed.csv")
X = Data["message"]
Y = Data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1103)


tfidf = TfidfVectorizer(ngram_range=(1, 2))
tfidf.fit_transform(X_train)
tfidf.fit_transform(X_test)
rfc = xgb.XGBClassifier(base_score=0.5, booster='gbtree', n_jobs=6,
                        objective='multi:softprob', verbosity=1)
# rrfc = xgb.XGBClassifier(n_jobs=7, objective='multi:softprob', verbosity=1)
param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15, 20, 25, 30, 35],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9],
    'eta': [0.05, 0.1, 0.3],
}
# rfc = GridSearchCV(estimator=rrfc, param_grid=param_grid, cv=10)
rfc.fit(tfidf.transform(X_train), y_train)
rfc_predict = rfc.predict(tfidf.transform(X_test))

print("ACCURACY SCORE:", metrics.accuracy_score(y_test, rfc_predict))
print("::::Confusion Matrix::::")
print(confusion_matrix(y_test, rfc_predict))
print("\n")

print(":::Classification Report:::")
print(classification_report(y_test, rfc_predict, target_names=['Class 1', 'Class 2', 'Class 3', "Class 4"]))
print("\n")

print(pd.crosstab(y_test, rfc_predict, rownames=["Orgnl"], colnames=['Predicted']))

print("===Test Accuracy===")
print("Accuracy Score::: ", metrics.accuracy_score(y_test, rfc_predict))

# print(rfc.best_params_)
# print(rfc.best_score_)
# print(rfc.best_estimator_)
# print(rfc.cv_results_)
# print(rfc.error_score)

class_names = ["-1", "0", "1", "2"]
disp = metrics.plot_confusion_matrix(rfc, tfidf.transform(X_test), y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
plt.show()
