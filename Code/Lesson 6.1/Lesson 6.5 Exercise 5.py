# Go to the IMDb Top Rated Movies page: https://www.imdb.com/chart/top/ .
# Inspect the page and identify the HTML elements that contain the required information for each movie (title, release year, runtime, IMDb rating, and movie URL).
# Write a Python script using a web scraping library such as BeautifulSoup to extract the information from the page.
# Store the scraped information in a data structure (e.g., a list of dictionaries) for further analysis or export to a file.
# If necessary, clean and format the extracted data (e.g., removing unnecessary characters, converting data types).
# Print or display the extracted data in a readable format.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

base_url = "https://www.imdb.com/chart/top/"

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
# create the headers dictionary with the user-agent
headers = {'User-Agent': user_agent}

page = requests.get(base_url, headers=headers) # grab the page
soup = BeautifulSoup(page.content, "html.parser") # create your beautiful soup object

# find all list elements
listItems = soup.find_all('li', {'class': 'ipc-metadata-list-summary-item'})

title = []
release_year = []
runtime = []
imdbrating = []
movieurl = []

for moviesList in listItems: #loop all  divs   
    innerdiv = moviesList.find_all('div', {'class': 'cli-title-metadata'}) 
    
    spans = innerdiv[0].find_all('span')
    release_yr = spans[0].text.strip()
    time = spans[1].text.strip()
    
    rating = ""
    if len(spans) > 2:
       rating = spans[2].text.strip()
       
    #scrap title   
    headings = moviesList.find_all('h3')
    
    headings_text = headings[0].text.strip().split(".")  #clean the text by removing the index  
    title.append(headings_text[1].strip())  
    
    #print(headings)
    #print("\n")
    #scrap movie url
    anchors = moviesList.find_all('a')
    murl = "https://www.imdb.com"+anchors[0].get('href')
    movieurl.append(murl)
    
    release_year.append(release_yr)
    runtime.append(time)
    imdbrating.append(rating)    
    
#create a dataframe    
df = pd.DataFrame({'Title': title, 'Release year': release_year, 'Runtime': runtime, 'IMDB rating': imdbrating, 'Movie URL': movieurl})
#to csv
csv_output_path = os.path.dirname(os.path.realpath(__file__))+"\\Output\\IMDB-data.csv"
df.to_csv(csv_output_path, index=False)
print(df)  
















