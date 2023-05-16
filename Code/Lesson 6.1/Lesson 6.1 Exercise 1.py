# Fetch the number of crimes committed during date November 21, 2022 , under CRIME & INVESTIGATIONS section
# Link : https://www.dailytrends.co.ke/
import requests
from bs4 import BeautifulSoup

from urllib.request import urlopen # use request for the python urllib library
from bs4 import BeautifulSoup

page = requests.get("https://www.dailytrends.co.ke") # grab the page

soup = BeautifulSoup(page.content, "html.parser") # create your beautiful soup object
crimescontainer = soup.select('[id=pencifeatured_cat_99142]') # fetch the conntainer having the crime news

magicdetail_boxes = crimescontainer[0].find_all('div', {"class": "magcat-detail"}) #find the inner boxes ; div elements

for magicdetailsbox in magicdetail_boxes:
     h3headings = magicdetailsbox.find('h3').get_text() # grab the h3 
     print(h3headings)

print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Number of crimes reported : ")
print(len(magicdetail_boxes))
print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")     