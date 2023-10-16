import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import style

style.use("ggplot")
import nltk

nltk.data.path.append("D:/Mini Project/Twitter using streamlit/nltk_data")
from nltk.stem import PorterStemmer
from nltk.tokenize import punkt, word_tokenize
from textblob import TextBlob

nltk.data.path.append("D:/Mini Project/Twitter using streamlit/nltk_data")
from nltk.corpus import stopwords

# nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words("english"))
import pickle

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC
from wordcloud import WordCloud


# Load the data
@st.cache_resource
def load_data():
    df = pd.read_csv("vaccination_tweets.csv")
    return df


df = load_data()


# Data preprocessing
def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


text_df = df.drop(
    [
        "id",
        "user_name",
        "user_location",
        "user_description",
        "user_created",
        "user_followers",
        "user_friends",
        "user_favourites",
        "user_verified",
        "date",
        "hashtags",
        "source",
        "retweets",
        "favorites",
        "is_retweet",
    ],
    axis=1,
)

text_df.text = text_df["text"].apply(data_processing)
text_df = text_df.drop_duplicates("text")
stemmer = PorterStemmer()


def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


text_df["text"] = text_df["text"].apply(lambda x: stemming(x))


def polarity(text):
    return TextBlob(text).sentiment.polarity


text_df["polarity"] = text_df["text"].apply(polarity)


def sentiment(label):
    if label < 0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label > 0:
        return "Positive"


text_df["sentiment"] = text_df["polarity"].apply(sentiment)

# Visualizations
st.title("Twitter Sentiment Analysis")
st.subheader("Data Analysis and Machine Learning Model")

# Display the data
st.subheader("The Data")
st.dataframe(df.head())

# Display data before processing
st.subheader("Unprocessed Data")
st.dataframe(df.head())

# Display processed data
st.subheader("Processed Data")
st.dataframe(text_df.head())

# Distribution of sentiments
st.subheader("Distribution of Sentiments")
fig = plt.figure(figsize=(5, 5))
sns.countplot(x="sentiment", data=text_df)
st.pyplot(fig)

# Word cloud for positive tweets
st.subheader("Word Cloud - Positive Tweets")
pos_tweets = text_df[text_df.sentiment == "Positive"]
text = " ".join([word for word in pos_tweets["text"]])
plt.figure(figsize=(20, 15), facecolor="None")
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most frequent words in positive tweets", fontsize=19)
st.pyplot(plt)

# Word cloud for negative tweets
st.subheader("Word Cloud - Negative Tweets")
neg_tweets = text_df[text_df.sentiment == "Negative"]
text = " ".join([word for word in neg_tweets["text"]])
plt.figure(figsize=(20, 15), facecolor="None")
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most frequent words in negative tweets", fontsize=19)
st.pyplot(plt)

# Word cloud for neutral tweets
st.subheader("Word Cloud - Neutral Tweets")
neutral_tweets = text_df[text_df.sentiment == "Neutral"]
text = " ".join([word for word in neutral_tweets["text"]])
plt.figure(figsize=(20, 15), facecolor="None")
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most frequent words in neutral tweets", fontsize=19)
st.pyplot(plt)

# Train the model
st.subheader("Train the Model")
vect = CountVectorizer(ngram_range=(1, 2)).fit(text_df["text"])
X = text_df["text"]
Y = text_df["sentiment"]
X = vect.transform(X)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)

# Model evaluation
st.subheader("Model Evaluation")
st.write("Accuracy Score:", accuracy_score(y_test, logreg_pred))
st.write("\nClassification Report:\n", classification_report(y_test, logreg_pred))
cm = confusion_matrix(y_test, logreg_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=logreg.classes_)
cmd.plot()
plt.title("Confusion Matrix")
st.pyplot(plt)

# Save the model as a pickle file
with open("model.pkl", "wb") as file:
    pickle.dump(logreg, file)

# Load the model from the pickle file
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Test the loaded model
loaded_pred = model.predict(x_test)
loaded_acc = accuracy_score(loaded_pred, y_test)
st.write("Loaded Model Accuracy:", loaded_acc)

st.subheader("Additional Model - Support Vector Machine (SVM)")
SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)
svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)

st.write("SVM Test Accuracy: {:.2f}%".format(svc_acc * 100))
st.write(confusion_matrix(y_test, svc_pred))
st.write("\n")
st.write(classification_report(y_test, svc_pred))

st.subheader("Grid Search for SVM")
param_grid = {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)

st.write("Best Parameters : ", grid.best_params_)
y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred, y_test)

st.write("Test Accuracy: {:.2f}%".format(logreg_acc * 100))
st.write(confusion_matrix(y_test, y_pred))
st.write("\n")
st.write(classification_report(y_test, y_pred))

if logreg_acc < 0:
    sentiment = "Negative"
elif logreg_acc == 0:
    sentiment = "Neutral"
else:
    sentiment = "Positive"

st.write("The Sentiment of the Dataset is : ", sentiment)
