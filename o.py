import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
import io
import missingno
import seaborn as sns
dataset=pd.read_csv("C:/Users/Um Ar/PycharmProjects/Internship-2/twitter_sentiment_data.csv")
dataset.head()
dataset.shape
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 43942):
    # Removing Hashtags
    review = re.sub(r'#', '', dataset['message'][i])
    # Removing Chines
    review = re.sub(r'[^\x00-\x7F]+', '', dataset['message'][i])
    # Removing Retweets
    review = re.sub(r'RT[\s]+', '', dataset['message'][i])
    # Removing HyperLinks
    review = re.sub(r'https?:\/\/\s+', '', dataset['message'][i])
    #selecting characters only
    review = re.sub('[^a-zA-Z]', ' ', dataset['message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:43942, 0].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
from sklearn import svm

#Create a svm Classifier
param = {"kernel": ['rbf']}

cclf = svm.SVC(kernel='linear', verbose=True) # Linear Kernel
clf = GridSearchCV(estimator=cclf, cv=2, param_grid=param, n_jobs=7, verbose=True)

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
