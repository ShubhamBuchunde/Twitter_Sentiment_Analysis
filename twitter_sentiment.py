# =============================================================================
# 1. Initializing tokens
# 2. Client authentication
# 3. Fetching Tweets
# 4. Loading TFIDF model
# 5. Preprocessing
# 6. Predicting sentiments of tweets
# 7. Plotting the result 
# 
# =============================================================================

import re
import pickle
import tweepy

from tweepy import OAuthHandler

#Initializing the keys(keys for auhentication the client)
consumer_key = "pq2SEdwR30gNUPDCtZAvLBEJE"
consumer_secret="Vc3euo3MMT8AUQE0vVZTioeJ13OAk2GFfQQ50lGy7SM4VNrTAV"
access_token="1105159507406053380-ifKwOnRYOaojuihg5NxYVfGP0LtQa2"
access_secret="Nc2MZTWA2r122KlQrM9Px05vRuTNYyRL11c5BM2QRveF6"


auth = OAuthHandler(consumer_key,consumer_secret)#wheter the application is twitter or not
auth.set_access_token(access_token,access_secret)#right to  fetch tweets from twitter
args = ['facebook']
api = tweepy.API(auth,timeout=10)

list_tweets = []

query = args[0]
if len(args)== 1:
    for status in tweepy.Cursor(api.search,q = query+ " -filter:retweets",lang = 'en',result_type='recent').items(100):
        list_tweets.append(status.text)

with open('tfidfmodel.pickle','rb') as f:
    vectorizer = pickle.load(f)
with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)
tot_pos = 0
tot_neg = 0

for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-z0-9]*$"," ",tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"could't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    sent = clf.predict(vectorizer.transform([tweet]).toarray())
    if sent[0] == 1:
        tot_pos += 1
    else:
        tot_neg +=1
        
#Plotting the bar chart 
import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive','Negative']
y_pos = np.arange(len(objects))    

plt.bar(y_pos,[tot_pos,tot_neg],alpha = 0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of positive and negative tweets')

plt.show()
    




