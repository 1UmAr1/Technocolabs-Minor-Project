import tweepy
import time
import pandas as pd
pd.set_option('display.max_colwidth', 1000)

# api key
api_key = "EmSGfrdY28btB3kEFlkayJGI5"
# api secret key
api_secret_key = "OmFNtqo8NyQBuQfDa9mL0L4TUiTkWeACGGpD3aUlvPRSICiFXM"
# access token
access_token = "931297071847448576-5w5I5uylTSvOF4t9DII78Yy3FL0cmyD"
# access token secret
access_token_secret = "hCihGWmkFEHo3jUV5ksLpi8PHIeeZxshXt84d95uEuXIg"

authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)


def get_related_tweets(text_query):
    # list to store tweets
    tweets_list = []
    # no of tweets
    count = 200
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