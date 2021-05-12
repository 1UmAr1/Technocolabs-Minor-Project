import tweepy
import time
import pandas as pd
pd.set_option('display.max_colwidth', 1000)

# api key
api_key = "4bZipg8rQIXulDt6iMYepZLM2"
# api secret key
api_secret_key = "Yy9b0t0vLZkx6kGfHEorOSAks3CpqgAIcHWUilBwPrRcQILZv2"
# access token
access_token = "931297071847448576-OMsxZaQsWhFWqrQ3bYwgc1oXVrFChGG"
# access token secret
access_token_secret = "0ucCqjYGPgeQ7QuR2pxQnBRhNraVRQkgGiSYQEjaAMA4g"

authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)


def get_related_tweets(text_query):
    # list to store tweets
    tweets_list = []
    # no of tweets
    count = 15000
    try:
        # Pulling individual tweets from query
        for tweet in api.search(q=text_query, count=count):
            print(tweet.text)
            # Adding to list that contains all tweets
            tweets_list.append({'created_at': tweet.created_at,
                                'tweet_id': tweet.id,
                                'tweet_text': tweet.text})
        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)
