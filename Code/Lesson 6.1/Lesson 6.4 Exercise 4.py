# Display the data obtained in question no. 2 as a DataFrame
# Link : https://nation.africa/kenya
import pandas as pd
import requests
from bs4 import BeautifulSoup


base_url = "https://nation.africa/kenya"
page = requests.get(base_url) # grab the page

soup = BeautifulSoup(page.content, "html.parser") # create your beautiful soup object

featuredteasers = soup.find_all('div', {"class": "teaser-image-left_summary"}) # fetch elements/objects in the teaser div
print(featuredteasers)

hold_our_teasers = [] #create a list
for teaser in featuredteasers: #loop all teaser divs
    hteser = teaser.find('h3') #find the h3 tag
    hold_our_teasers.append(hteser.text.strip()) #strip the h3 tag to get the exact text and add it on a list

#create the dataframe using pandas
columns = ["Teasers"]
df = pd.DataFrame(hold_our_teasers,columns=columns)
#print(df)
