from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load real sentiment dataset
dataset = load_dataset("tweet_eval", "sentiment")

texts = dataset["train"]["text"]
labels = dataset["train"]["label"]  # 0=negative, 1=neutral, 2=positive

# Convert text to numbers (better than CountVectorizer)
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

print("Model trained on real sentiment dataset ✅")

import pickle
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

