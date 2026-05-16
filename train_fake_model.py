import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset (we’ll use a simple CSV)
# Download this dataset and place in same folder:
# https://raw.githubusercontent.com/clmentbisaillon/fake-and-real-news-dataset/master/Fake.csv
# https://raw.githubusercontent.com/clmentbisaillon/fake-and-real-news-dataset/master/True.csv

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0   # fake
true["label"] = 1   # real

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

texts = data["title"]
labels = data["label"]

# Convert text → numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Fake News Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("fake_model.pkl", "wb"))
pickle.dump(vectorizer, open("fake_vectorizer.pkl", "wb"))

print("Fake news model saved ✅")
