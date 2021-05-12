import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

Data = pd.read_csv("First_processed.csv")
X = Data["message"]
Y = Data["sentiment"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1103)

# Splitting the data into validation
X_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1103)

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=12000, use_idf=True)
tfidf.fit_transform(X_train)
tfidf.fit_transform(x_val)
logreg = OneVsRestClassifier(SVC())
logreg.fit(tfidf.transform(X_train), y_train)

rfc_predict = logreg.predict(tfidf.transform(x_val))
print("ACCURACY SCORE:", metrics.accuracy_score(y_val, rfc_predict))
print("::::Confusion Matrix::::")
print(confusion_matrix(y_val, rfc_predict))
print("\n")

print(":::Classification Report:::")
print(classification_report(y_val, rfc_predict, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))
print("\n")

print(pd.crosstab(y_val, rfc_predict, rownames=["Orgnl"], colnames=['Predicted']))


