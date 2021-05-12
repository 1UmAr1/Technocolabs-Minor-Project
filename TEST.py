import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import io
import missingno
import seaborn as sns
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("twitter_sentiment_data.csv")
dataset.head()

dataset.isna().sum()

corpus = []
for i in range(0, 43942):
    review = re.sub(r'@[A-Za-z0-9]+', '', dataset['message'][i])
    # Removing Hashtags
    review = re.sub(r'#', '', dataset['message'][i])
    # Removing Chines
    review = re.sub(r'[^\x00-\x7F]+', '', dataset['message'][i])
    # Removing Retweets
    review = re.sub(r'RT[\s]+', '', dataset['message'][i])
    # Removing HyperLinks
    review = re.sub(r'https?:\/\/\s+', '', dataset['message'][i])
    # Removing numeric values
    review = re.sub(r'\d+', '', dataset['message'][i])
    review = re.sub(r'aa[A-Za-z0-9]+', '', dataset['message'][i])
    review = re.sub(r'zz[A-Za-z0-9]+', '', dataset['message'][i])
    review = review.lower()
    review = re.sub('[^a-zA-Z]', ' ', dataset['message'][i])
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

len(corpus)

cv = TfidfVectorizer(max_features=10000, use_idf=True, ngram_range=(1, 3))
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:43942, 0].values
print(x)
dataset['word_counts'] = dataset['message'].str.split().str.len()
dataset["Text Length"] = dataset["message"].str.len()
dataset.groupby('sentiment')['word_counts'].mean()
frame = pd.DataFrame(x)

# frame["sentiment"] = y
frame["word_counts"] = dataset['word_counts']
frame["Text Length"] = dataset["Text Length"]
print(frame.columns)
# frame.to_csv("Data.csv")
# frame.drop('sentiment', axis=1)
x_train, x_test, y_train, y_test = train_test_split(frame, y, test_size=0.20, random_state=0)

rfc = RandomForestClassifier(n_jobs=7, verbose=True)
# LogisticRegression(verbose=True, max_iter=2000, n_jobs=7)
rfc.fit(x_train, y_train)

rfc_predict = rfc.predict(x_test)
print("ACCURACY SCORE:", metrics.accuracy_score(y_test, rfc_predict))
print("::::Confusion Matrix::::")
print(confusion_matrix(y_test, rfc_predict))
print("\n")

print(":::Classification Report:::")
print(classification_report(y_test, rfc_predict, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))
print("\n")

print(pd.crosstab(y_test, rfc_predict, rownames=["Orgnl"], colnames=['Predicted']))

