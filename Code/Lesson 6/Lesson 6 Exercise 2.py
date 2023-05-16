#Write a program that grabs the full HTML from the page yahoo finance (https://finance.yahoo.com/).
#Using Beautiful Soup, print out a list of all the links on the page by looking for HTML tags with the name a and retrieving the value taken on by the href attribute of each tag.
#Display the current value of the BTC-USD price

import requests
from bs4 import BeautifulSoup

base_url = "https://finance.yahoo.com"

page = requests.get(base_url) # grab the page

soup = BeautifulSoup(page.content, "html.parser") # create your beautiful soup object

#loop all links
for links in soup.find_all("a"):   
     print ("Found the URL:", links['href'])

#Display the current value of the BTC-USD price
pagebtc = requests.get(base_url+"/crypto")
soup = BeautifulSoup(pagebtc.content, "html.parser")

for item in soup.select('.simpTblRow'):	#.select() method which is used to run a CSS selector against a parsed document and return all the matching elements i.e for each row do a loop
         label = item.select('[aria-label=Symbol]')[0].get_text()
         value = item.select('[aria-label*=Price]')[0].get_text()
         if(label == "BTC-USD"):
            print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(item.select('[aria-label=Symbol]')[0].get_text() +"  "+ value)
            print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")


 

