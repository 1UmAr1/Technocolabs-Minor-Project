import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


Data = pd.read_csv("First_processed.csv")
X = Data["message"]
Y = Data["sentiment"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1103)

# Splitting the data into validation
X_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1103)


tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, use_idf=True)
tfidf.fit_transform(X_train)
tfidf.fit_transform(x_val)

rfc = KNeighborsClassifier(n_jobs=7)
param_grid = {
    'max_features': ["auto", "log2", "sqrt"],

}
X_train = tfidf.transform(X_train)
MinMaxScaler = preprocessing.MaxAbsScaler()

X_train = MinMaxScaler.fit_transform(X_train)

x_val = tfidf.transform(x_val)
x_val = MinMaxScaler.fit_transform(x_val)

# rfc = GridSearchCV(estimator=rrfc, param_grid=param_grid, cv=5)
rfc.fit(X_train, y_train)
rfc_predict = rfc.predict(x_val)
print("ACCURACY SCORE:", metrics.accuracy_score(y_val, rfc_predict))
print("::::Confusion Matrix::::")
print(confusion_matrix(y_val, rfc_predict))
print("\n")

print(":::Classification Report:::")
print(classification_report(y_val, rfc_predict, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))
print("\n")

print(pd.crosstab(y_test, rfc_predict, rownames=["Orgnl"], colnames=['Predicted']))
