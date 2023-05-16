#1 ). Using the tripadvisor website , webscrap for the feedback by customers while visting the spring gardens in Australia
# Link : https://www.tripadvisor.com/Articles-l0Mk5FCTGTRE-Where_to_find_spring_flowers_in_australia.html  
# Create a dataframe from the webscrapped data and save to a csv , the title of the place , period of travel , sentiments
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

page_url = "https://www.tripadvisor.com/Articles-l0Mk5FCTGTRE-Where_to_find_spring_flowers_in_australia.html" #url for tripadvisor

#add headers to trick trip advisor server it is a web browser accessing the page
user_agent = (
 {
 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
 'Accept-Language': 'en-US, en;'
 }
)
page = requests.get(page_url,headers = user_agent) # grab the page

#print(page.content)
soup = BeautifulSoup(page.content, "html.parser") # create your beautiful soup object

title_of_the_place = []
for title in soup.findAll('h3',{'class':'biGQs'}):
    title_of_the_place.append(title.text.strip())
    #print(title_of_the_place)
    
period_of_travel = []
for period in soup.findAll('li',{'class':'YprNL'}):
    period_text = period.text.strip();   
    #check of key word when to indicate is a time period
    slice_word_when = period_text[0:4] #used to slice the text from index 0, to the end index
    filtered_text = period_text[6:]  # empty end index meaning the entrie from index 6
    if(slice_word_when == 'When'):
        period_of_travel.append(filtered_text)        
        #print(period_of_travel)
    
sentiments = []
for sentiment in soup.findAll('blockquote',{'class':'biGQs'}):
    sentiments_raw = sentiment.text.strip() 
    sentiments_cleaned = sentiments_raw.encode(encoding="ascii",errors="ignore")
    sentiments.append(sentiments_cleaned)
    #print(sentiments)

df = pd.DataFrame({'Title': title_of_the_place,'Period': period_of_travel,'Sentiment': sentiments}) # create the data frame from the list

df.to_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\TripAdvisorScrappedData.csv") # save to csv file

print(df)

