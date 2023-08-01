# importing the Python libraries necessary
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from wordcloud import WordCloud 
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from yellowbrick.classifier import PrecisionRecallCurve # pip install yellowbrick 
import warnings
warnings.filterwarnings("ignore")


# read the csv file
df= pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Financial Data.csv")

# get an idea of how many positive, negative and neutral sentences
print(df['Sentiment'].value_counts())
sns.countplot(x='Sentiment',data=df);

# start with cleaning and pre-processing the text
stop_words = set(stopwords.words("english"))
df["Sentence"] = df["Sentence"].str.replace("\d","")
def cleaner(data):     
    # Tokens
    tokens = word_tokenize(str(data).replace("'", "").lower())      
    # Remove Puncs
    without_punc = [w for w in tokens if w.isalpha()]     
    # Stopwords
    without_sw = [t for t in without_punc if t not in stop_words]     
    # Lemmatize
    text_len = [WordNetLemmatizer().lemmatize(t) for t in without_sw]
    # Stem
    text_cleaned = [PorterStemmer().stem(w) for w in text_len]     
    return " ".join(text_cleaned)

df["Sentence"] = df["Sentence"].apply(cleaner)

# filter out the rare words that are used in the sentences
rare_words = pd.Series(" ".join(df["Sentence"]).split()).value_counts()
rare_words = rare_words[rare_words <= 2]
 
df["Sentence"] = df["Sentence"].apply(lambda x: " ".join([i for i in x.split() if i not in rare_words.index]))


plt.figure(figsize=(16,12))
wordcloud = WordCloud(background_color="black",max_words=500, width=1500, height=1000).generate(' '.join(df['Sentence']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#custom encoder
target = []
for i in df["Sentiment"]:
    if i == "positive":
        target.append(1)
    elif i == "neutral":
        target.append(0)
    else:
        target.append(-1)
df["Target"] = target

X = df["Sentence"]
y = df["Target"] 

# divide the data set into two parts
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25,random_state= 42,stratify=y)


# convert textual data into a numerical representation called a "bag of words" representation
vt = CountVectorizer(analyzer="word")
X_train_count = vt.fit_transform(X_train)
X_test_count = vt.transform(X_test)


# execute the Multinomial Naïve Bayes algorithm on the training data
nb_model = MultinomialNB(force_alpha=True)
nb_model.fit(X_train_count,y_train)


# evaluate the performance of the trained MNB algorithm
nb_pred = nb_model.predict(X_test_count)
nb_train_pred = nb_model.predict(X_train_count)
print("X Test")
print(classification_report(y_test,nb_pred))
print("X Train")
print(classification_report(y_train,nb_train_pred))


# visualise results in the form of a heat map 
plt.figure(figsize=(8,8))
sns.heatmap(confusion_matrix(y_test,nb_pred),annot = True,fmt = "d")


# visualise  results using a precision-recall curve
viz = PrecisionRecallCurve(MultinomialNB(),classes=nb_model.classes_,per_class=True,cmap="Set1")
viz.fit(X_train_count,y_train)
viz.score(X_test_count, y_test) 
viz.show();















































