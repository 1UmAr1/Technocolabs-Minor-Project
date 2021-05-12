import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
Data = pd.read_csv("First_processed.csv")
X = Data["message"]
Y = Data["sentiment"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1103)

# Splitting the data into validation
X_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1103)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, use_idf=True)
tfidf.fit_transform(X_train)
tfidf.fit_transform(x_val)

X_train = tfidf.transform(X_train)
MinMaxScaler = preprocessing.Normalizer()

X_train = MinMaxScaler.fit_transform(X_train)

x_val = tfidf.transform(x_val)
x_val = MinMaxScaler.fit_transform(x_val)

grid = {"C": np.logspace(-1, -3, 3, 7, 9), "penalty": ['l2']}# l1 lasso l2 ridge
logreg = LogisticRegression(n_jobs=7, max_iter=4000, multi_class='ovr')
logreg_cv = GridSearchCV(logreg, grid, cv=5, verbose=True)
# X_train["word_count"] = Data["word_counts"]
# X_test["word_count"] = Data["word_counts"]
# X_train["Text Length"] = Data["Text Length"]
# X_test["Text Length"] = Data["Text Length"]
logreg.fit(X_train, y_train)
# print(logreg_cv.best_score_)
# print(logreg_cv.best_params_)
rfc_predict = logreg.predict(x_val)
print("ACCURACY SCORE:", metrics.accuracy_score(y_val, rfc_predict))
print("::::Confusion Matrix::::")
print(confusion_matrix(y_val, rfc_predict))
print("\n")

print(":::Classification Report:::")
print(classification_report(y_val, rfc_predict, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))
print("\n")

print(pd.crosstab(y_val, rfc_predict, rownames=["Orgnl"], colnames=['Predicted']))


