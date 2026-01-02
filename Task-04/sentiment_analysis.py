import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Load dataset
df = pd.read_csv(
    "P:\\Prodigy\\Task-04\\twitter_training.csv",
    header=None,
    names=["id", "entity", "sentiment", "tweet"]
)


df.head()
df.info()

#clean data 
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_tweet"] = df["tweet"].astype(str).apply(clean_text)


# Visualize sentiment distribution
df["sentiment"].value_counts().plot(kind="bar")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution in Twitter Data")
plt.show()


# Visualize sentiment by entity
top_entities = df["entity"].value_counts().head(10).index

entity_sentiment = df[df["entity"].isin(top_entities)]
pd.crosstab(entity_sentiment["entity"], entity_sentiment["sentiment"]).plot(
    kind="bar", figsize=(10,6)
)

plt.title("Sentiment by Top Entities")
plt.xlabel("Entity")
plt.ylabel("Count")
plt.show()
