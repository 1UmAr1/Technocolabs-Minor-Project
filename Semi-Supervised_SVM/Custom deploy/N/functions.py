import pandas
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd


def create_features_from_df(Data):
    corpus = []
    for i in range(0, len(Data)):
        # Removing Hashtags
        review = re.sub(r'#', '', Data['tweet_text'][i])
        # Removing Chines
        review = re.sub(r'[^\x00-\x7F]+', '', Data['tweet_text'][i])
        # Removing Retweets
        review = re.sub(r'RT[\s]+', '', Data['tweet_text'][i])
        # Removing HyperLinks
        review = re.sub(r'https?:\/\/\s+', '', Data['tweet_text'][i])
        # selecting characters only
        review = re.sub('[^a-zA-Z]', ' ', Data['tweet_text'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

