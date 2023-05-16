# Using K24Tv New , Extract titles of the featured bulletings/teasers on the homepage
# Link : https://nation.africa/kenya

import requests
from bs4 import BeautifulSoup

from urllib.request import urlopen # use request for the python urllib library
from bs4 import BeautifulSoup
base_url = "https://nation.africa/kenya"
page = requests.get(base_url) # grab the page

soup = BeautifulSoup(page.content, "html.parser") # create your beautiful soup object

featuredteasersright = soup.find_all('div', {"class": "teaser-image-right_summary"}) # fetch elements/objects in the teaser div
#featuredteaserleft = soup.find_all('div', {"class": "teaser-image-left_summary"}) # fetch elements/objects in the teaser div

for teaser in featuredteasersright: #loop all teaser divs
    hteser = teaser.find('h3')  
    print(hteser.text.strip())   # use find to get a single instance of header 3