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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
from scikitplot.metrics import plot_confusion_matrix #pip install scikit-plot
import os


# Read the training data and validation data
df_train = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\sentiments-nlp\\train.txt",delimiter=';',names=['text','label'])
df_val = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\sentiments-nlp\\val.txt",delimiter=';',names=['text','label'])

df_train = df_train.sample(n = 100, random_state = 20)
df_val = df_val.sample(n = 100, random_state = 20)

# Concatenate these two data frames
df = pd.concat([df_train,df_val])
df.reset_index(inplace=True,drop=True)


# Check for the various target labels in our dataset using seaborn.
df['label'] = df['label'].astype('category') # If the data type is "float64" or "int64," you can convert it to the categorical data type
sns.countplot(data=df, x='label')
plt.show()


# Create a custom encoder to convert categorical target labels to numerical form, i.e. (0 and 1)
def custom_encoder(df):
    df.replace(to_replace ="surprise", value =1, inplace=True)
    df.replace(to_replace ="love", value =1, inplace=True)
    df.replace(to_replace ="joy", value =1, inplace=True)
    df.replace(to_replace ="fear", value =0, inplace=True)
    df.replace(to_replace ="anger", value =0, inplace=True)
    df.replace(to_replace ="sadness", value =0, inplace=True)
custom_encoder(df['label'])
sns.countplot(data=df, x='label')
plt.show()



# Object of WordNetLemmatizer
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


# Create a Word Cloud. It is a data visualization technique used to depict text in such a way that, the more frequent words appear enlarged as compared to less frequent words.
plt.rcParams['figure.figsize'] = 20,8
word_cloud = ""
for row in corpus:
    for word in row:
        word_cloud+=" ".join(word)
wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
plt.imshow(wordcloud)



#onvert the text data into vectors, by fitting and transforming the corpus that we have created.
cv = CountVectorizer(ngram_range=(1,2))
traindata = cv.fit_transform(corpus)
X = traindata
y = df.label


# Create a dictionary, “parameters” which will contain the values of different hyperparameters
parameters = {'max_features': ('auto','sqrt'),
             'n_estimators': [500, 1000, 1500],
             'max_depth': [5, 10, None],
             'min_samples_split': [5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10],
             'bootstrap': [True, False]}

# Fit the data into the grid search and view the best parameter using the “best_params_” attribute of GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(),parameters,cv=5,return_train_score=True,n_jobs=-1)
grid_search.fit(X,y)
grid_search.best_params_


# View all the models and their respective parameters, mean test score and rank as  GridSearchCV stores all the results in the cv_results_ attribute.
# for i in range(432):
#     print('Parameters: ',grid_search.cv_results_['params'][i])
#     print('Mean Test Score: ',grid_search.cv_results_['mean_test_score'][i])
#     print('Rank: ',grid_search.cv_results_['rank_test_score'][i])


# Choose the best parameters obtained from GridSearchCV and create a final random forest classifier model and then train our new model.
rfc = RandomForestClassifier(max_features=grid_search.best_params_['max_features'],
                                      max_depth=grid_search.best_params_['max_depth'],
                                      n_estimators=grid_search.best_params_['n_estimators'],
                                      min_samples_split=grid_search.best_params_['min_samples_split'],
                                      min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                      bootstrap=grid_search.best_params_['bootstrap'])
rfc.fit(X,y)



# Read the test data and perform the same transformations we did on training data
test_df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\sentiments-nlp\\test.txt",delimiter=';',names=['text','label'])
X_test,y_test = test_df.text,test_df.label
#encode the labels into two classes , 0 and 1
test_df = custom_encoder(y_test)
#pre-processing of text
test_corpus = text_transformation(X_test)
#convert text data into vectors
testdata = cv.transform(test_corpus)
#predict the target
predictions = rfc.predict(testdata)



# Evaluate our model using various metrics such as Accuracy Score, Precision Score, Recall Score, Confusion Matrix and create a roc curve to visualize how our model performed
plt.rcParams['figure.figsize'] = 10,5
plot_confusion_matrix(y_test,predictions)
acc_score = accuracy_score(y_test,predictions)
pre_score = precision_score(y_test,predictions)
rec_score = recall_score(y_test,predictions)
print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score)
print('Recall_score: ',rec_score)
print("-"*50)
cr = classification_report(y_test,predictions)
print(cr)



# Find the probability of the class using the predict_proba() method of Random Forest Classifier and then we will plot the roc curve
predictions_probability = rfc.predict_proba(testdata)
fpr,tpr,thresholds = roc_curve(y_test,predictions_probability[:,1])
plt.plot(fpr,tpr)
plt.plot([0,1])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()



# Predict for Custom Input
def expression_check(prediction_input):
    if prediction_input == 0:
        print("Input statement has Negative Sentiment.")
    elif prediction_input == 1:
        print("Input statement has Positive Sentiment.")
    else:
        print("Invalid Statement.")
# function to take the input statement and perform the same transformations we did earlier
def sentiment_predictor(input):
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = rfc.predict(transformed_input)
    expression_check(prediction)
    
input1 = ["Sometimes I just want to punch someone in the face."]
input2 = ["I bought a new phone and it's so good."]

sentiment_predictor(input1)
sentiment_predictor(input2)








































































































