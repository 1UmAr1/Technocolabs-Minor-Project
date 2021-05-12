import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,\
    RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix,\
    f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

Data = pd.read_csv("C:/Users/Um Ar/PycharmProjects/Internship-2/First_processed.csv")
val = pd.read_csv("C:/Users/Um Ar/PycharmProjects/Internship-2/SEMI_PREDICTED.csv")

print(val.columns)
print(Data.columns)

val["word_counts"] = val["message"].str.split().str.len()
val["Text Length"] = val["message"].str.len()
val.groupby("sentiment")['word_counts'].mean()
Data = pd.concat([Data, val])
print(Data.columns)
Y = Data["sentiment"]
X = Data
X = X.drop('sentiment', axis=1)
print(X.columns)
print(Y)
# X = Data["message"]
# Y = Data["sentiment"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1103)
train_words = X_train["word_counts"]
train_text = X_train["Text Length"]

train_words = pd.DataFrame(train_words)
train_text = pd.DataFrame(train_text)

train_lengths = pd.concat([train_words, train_text], axis=1)

test_words = X_test["word_counts"]
test_text = X_test["Text Length"]

test_words = pd.DataFrame(test_words)
test_text = pd.DataFrame(test_text)

test_lengths = pd.concat([test_words, test_text], axis=1)

X_train = X_train["message"]
X_test = X_test["message"]
print(X_train)
print(X_test)
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, use_idf=True)
tfidf.fit_transform(X_train)
tfidf.fit_transform(X_test)

MinMaxScaler = preprocessing.Normalizer()
X_train = tfidf.transform(X_train)
X_train = MinMaxScaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

X_test = tfidf.transform(X_test)
X_test = MinMaxScaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)


print(X_train.columns)
print(X_test)

print(train_lengths.columns)

X_train = pd.concat([X_train.reset_index(drop=True), train_lengths.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), test_lengths.reset_index(drop=True)], axis=1)
param_grid = {"C": [10],
              "gamma": [1],
              "kernel": ['rbf']}


ssvm = SVC(verbose=True)
svm = GridSearchCV(estimator=ssvm, param_grid=param_grid, n_jobs=7, refit=True, cv=2, verbose=True)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
print("ACCURACY SCORE:", metrics.accuracy_score(y_test, predictions))
print("::::Confusion Matrix::::")
print(confusion_matrix(y_test, predictions))
print("\n")

print(":::Classification Report:::")
print(classification_report(y_test, predictions, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))
print("\n")

print(pd.crosstab(y_test, predictions, rownames=["Orgnl"], colnames=['Predicted']))
class_names = ["-1", "0", "1", "2"]
disp = metrics.plot_confusion_matrix(svm, tfidf.transform(X_test), y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
plt.show()


# print(grid.best_score_)
# print(grid.best_params_)
# print(grid.error_score)

