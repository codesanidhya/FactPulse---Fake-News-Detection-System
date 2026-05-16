import requests
import pickle
import csv

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

API_KEY = "9a099bc4bd554276820695f620aa73f4"

query = input("Enter topic to search news: ")

url = f"https://newsapi.org/v2/everything?q=\"{query}\"&language=en&sortBy=relevancy&pageSize=10&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()
articles = data.get("articles", [])

with open("news_sentiment.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Headline", "Sentiment"])

    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        source = article['source']['name']

        if not title or not description:
            continue

        text = title + " " + description
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        if prediction == 2:
            mood = "Positive"
        elif prediction == 0:
            mood = "Negative"
        else:
            mood = "Neutral"

        writer.writerow([title, mood])

        print(title)
        print(description)
        print("Source:", source)
        print("Sentiment:", mood)
        print("-" * 40)

print("✅ Done! Data saved to news_sentiment.csv")
