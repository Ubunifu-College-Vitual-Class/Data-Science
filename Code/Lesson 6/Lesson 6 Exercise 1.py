# Write a program that grabs the full HTML from the following URL
# url = https://en.wikipedia.org/wiki/Main_Page
# Display the text from the featured article
from bs4 import BeautifulSoup

from urllib.request import urlopen

url = "https://en.wikipedia.org/wiki/Main_Page"

html_page = urlopen(url) # It is capable of retrieving URLs with a variety of protocols. 

html_text = html_page.read().decode("utf-8") # used to convert bytes to string object

soup = BeautifulSoup(html_text, "html.parser")# create your beautiful soup object

scrappedata = soup.find(id="mp-tfa") # get the div that displays the featured article

featured_artcle = soup.find("p") # scrap the paragraph

print(featured_artcle)# display


