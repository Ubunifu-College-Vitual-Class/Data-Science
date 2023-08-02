
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

from parsel import Selector # parsel is a Python library that provides a flexible and efficient way to extract data from web pages using XPath or CSS selectors.
import json # to handle json structures data

base_url = "https://syndication.twitter.com/srv/timeline-profile/screen-name/K24tv"
# "https://syndication.twitter.com/srv/timeline-profile/screen-name/HonMoses_Kuria"

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

print(tweets[0]["text"])




























