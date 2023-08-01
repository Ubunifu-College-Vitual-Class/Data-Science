import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud # pip install wordcloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
from scikitplot.metrics import plot_confusion_matrix #pip install scikit-plot
import os


# Read the training data and validation data
df_train = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\sentiments-nlp\\train.txt",delimiter=';',names=['text','label'])
df_val = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\sentiments-nlp\\val.txt",delimiter=';',names=['text','label'])

df_train = df_train.sample(n = 1500, random_state = 20)
df_val = df_val.sample(n = 1500, random_state = 20)

# Concatenate these two data frames
df = pd.concat([df_train,df_val])
df.reset_index(inplace=True,drop=True)


# # Check for the various target labels in our dataset using seaborn.
# df['label'] = df['label'].astype('category') # If the data type is "float64" or "int64," you can convert it to the categorical data type
# sns.countplot(data=df, x='label')
# plt.show()


# Create a custom encoder to convert categorical target labels to numerical form, i.e. (0 and 1)
# def custom_encoder(df):
#     df.replace(to_replace ="surprise", value =1, inplace=True)
#     df.replace(to_replace ="love", value =1, inplace=True)
#     df.replace(to_replace ="joy", value =1, inplace=True)
#     df.replace(to_replace ="fear", value =0, inplace=True)
#     df.replace(to_replace ="anger", value =0, inplace=True)
#     df.replace(to_replace ="sadness", value =0, inplace=True)
# custom_encoder(df['label'])
# sns.countplot(data=df, x='label')
# plt.show()



# # Object of WordNetLemmatizer
# nltk.download('stopwords') # download additional resources needed for certain NLTK functionalities. 
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lm = WordNetLemmatizer()
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus


corpus = text_transformation(df['text'])


# # Create a Word Cloud. It is a data visualization technique used to depict text in such a way that, the more frequent words appear enlarged as compared to less frequent words.
# plt.rcParams['figure.figsize'] = 20,8
# word_cloud = ""
# for row in corpus:
#     for word in row:
#         word_cloud+=" ".join(word)
# wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
# plt.imshow(wordcloud)



# #onvert the text data into vectors, by fitting and transforming the corpus that we have created.
cv = CountVectorizer(ngram_range=(1,2))
traindata = cv.fit_transform(corpus)
X = traindata
y = df.label

# Step 1: Split the data into training and validation sets
from sklearn.model_selection import train_test_split
from sklearn.metrics import  recall_score, f1_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 2: Train a LogisticRegression classifier
logreg_classifier = LogisticRegression(random_state=42, max_iter=1000)

logreg_classifier.fit(X_train, y_train)

# Step 3: Make predictions on the validation data
y_pred = logreg_classifier.predict(X_val)

# Step 4: Evaluate the model's performance
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=1)
confusion_mat = confusion_matrix(y_val, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

#After evaluation, you can use this trained model to predict the sentiment of new texts.

new_text = ["I hate this movie"]

new_text_transformed = text_transformation(new_text)
new_text_vectorized = cv.transform(new_text_transformed)
prediction = logreg_classifier.predict(new_text_vectorized)


print("Predicted Sentiment:", prediction[0])






























































































