
import pandas as pd
import requests
import os
import seaborn as sns
from parsel import Selector # parsel is a Python library that provides a flexible and efficient way to extract data from web pages using XPath or CSS selectors.
import json # to handle json structures data


handle = "HonMoses_Kuria" #HonMoses_Kuria #K24tv

base_url = "https://syndication.twitter.com/srv/timeline-profile/screen-name/"+ handle


user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
# create the headers dictionary with the user-agent
headers = {'User-Agent': user_agent}

page = requests.get(base_url, headers=headers) # grab the page
page_content = page.content.decode('utf-8')  # Decode the content from bytes to string

sel = Selector(text=page_content)  # Use the decoded page content as input to Selector

# find data cache:
data = json.loads(sel.css("script#__NEXT_DATA__::text").get())
# parse tweet data from data cache JSON:
tweet_data = data["props"]["pageProps"]["timeline"]["entries"]
tweets = [tweet["content"]["tweet"] for tweet in tweet_data]

columns = ["Tweet"]
tweetlist = []
for tweetdata in tweets:   
    tweetlist.append(tweetdata["text"]) 
    
tweets_df = pd.DataFrame(tweetlist, columns=columns)    
#print(tweets_df)

afintxt = os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\AFINN-111.txt"

afinnfile = open(afintxt)

scores = {}  # initialize an empty dictionary
for line in afinnfile:
# the file is tab-delimited. "\t" means "tab character"
    term, score = line.split("\t")
# convert the score to an integer.
    scores[term] = int(score)  

# function to calculate sentiment score for a given tweet
def calculate_sentiment_score(tweet):
# split the tweet into individual words    
    words = tweet.split()
# initialize sentiment score for the tweet    
    sentiment_score = 0 
    for word in words:
        if word in scores:
            sentiment_score += scores[word]
    return sentiment_score

# add a new column 'SentimentScore' to the DataFrame
tweets_df['SentimentScore'] = tweets_df['Tweet'].apply(calculate_sentiment_score)

#custom encoder
target = []
for i in tweets_df["SentimentScore"]:
    if i > 0 :
        target.append("positive sentiment")
    elif i == 0:
        target.append("neutral")
    else:
        target.append("negative sentiment")
tweets_df["Target"] = target


# get an idea of how many positive, negative and neutral sentences
print(tweets_df['Target'].value_counts())
sns.countplot(x='Target',data=tweets_df);

print(tweets_df)






























