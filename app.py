from flask import Flask, render_template, request, jsonify
import requests
import pickle
import subprocess

app = Flask(__name__)

# -------- Load ML Models --------
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

fake_model = pickle.load(open("fake_model.pkl", "rb"))
fake_vectorizer = pickle.load(open("fake_vectorizer.pkl", "rb"))

NEWS_API_KEY = "9a099bc4bd554276820695f620aa73f4"

# -------- Cache --------
ollama_cache = {}

# -------- Ollama --------
def get_ai_summary(text):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", f"Summarize in 2 simple lines:\n{text}"],
            capture_output=True, text=True, timeout=25
        )
        return result.stdout.strip()
    except:
        return "Summary not available."

def get_ai_explanation(title, fake_label):
    try:
        prompt = f"Explain in detail with full context and reasoning in 4-5 lines why this news might be {fake_label}: {title}"
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True, text=True, timeout=25
        )
        return result.stdout.strip()
    except:
        return "Explanation not available."

# ⭐ FRONT PAGE
@app.route("/")
def front():
    return render_template("front.html")

# ⭐ SEARCH RESULTS PAGE
@app.route("/search", methods=["POST"])
def search():
    news_data = []
    query = request.form["topic"]

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])

    for article in articles[:10]:
        title = article.get("title", "No title")

        # Sentiment
        X = vectorizer.transform([title])
        mood = ["Negative", "Neutral", "Positive"][model.predict(X)[0]]

        # Fake / Real
        X_fake = fake_vectorizer.transform([title])
        fake_pred = fake_model.predict(X_fake)[0]
        fake_label = "Fake" if fake_pred == 0 else "Real"
        fake_prob = round(fake_model.predict_proba(X_fake)[0][1] * 100, 2)

        news_data.append({
            "title": title,
            "desc": article.get("description", ""),
            "url": article.get("url", "#"),
            "mood": mood,
            "fake_label": fake_label,
            "fake_prob": fake_prob
        })

    return render_template("index.html", news_data=news_data)

# ⭐ TRENDING NEWS
@app.route("/trending")
def trending():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=6&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])

    trending_news = []
    for a in articles:
        trending_news.append({
            "title": a.get("title"),
            "image": a.get("urlToImage") or "https://picsum.photos/400/300",
            "source": a.get("source", {}).get("name"),
            "url": a.get("url")
        })

    return jsonify(trending_news)

# ⭐ DASHBOARD DETAILS
@app.route("/details", methods=["POST"])
def details():
    data = request.json
    title = data["title"]
    desc = data["desc"]
    fake_label = data["fake_label"]

    if title in ollama_cache:
        return jsonify(ollama_cache[title])

    summary = get_ai_summary(desc)
    explanation = get_ai_explanation(title, fake_label)

    result = {"summary": summary, "explanation": explanation}
    ollama_cache[title] = result

    return jsonify(result)

# ⭐ REDDIT
@app.route("/reddit", methods=["POST"])
def reddit():
    data = request.json
    title = data["title"]

    try:
        search_url = f"https://www.reddit.com/search.json?q={title}&limit=5"
        headers = {"User-Agent": "factpulse-app"}

        response = requests.get(search_url, headers=headers)
        data = response.json()

        posts = data.get("data", {}).get("children", [])

        reddit_posts = []

        for post in posts:
            post_data = post.get("data", {})
            reddit_posts.append({
                "text": post_data.get("title"),
                "url": "https://reddit.com" + post_data.get("permalink", "")
            })

        return jsonify(reddit_posts)

    except:
        return jsonify([])

# ⭐ NEW: INDIA NEWS (LAST 24 HOURS)
@app.route("/india_news")
def india_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&pageSize=6&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])

    news = []

    for a in articles:
        title = a.get("title", "")
        desc = a.get("description", "")

        # Sentiment
        X = vectorizer.transform([title])
        mood = ["Negative", "Neutral", "Positive"][model.predict(X)[0]]

        # Fake detection
        X_fake = fake_vectorizer.transform([title])
        fake_pred = fake_model.predict(X_fake)[0]
        fake_label = "Fake" if fake_pred == 0 else "Real"
        fake_prob = round(fake_model.predict_proba(X_fake)[0][1] * 100, 2)

        news.append({
            "title": title,
            "desc": desc,
            "url": a.get("url", "#"),
            "mood": mood,
            "fake_label": fake_label,
            "fake_prob": fake_prob
        })

    return jsonify(news)

if __name__ == "__main__":
    app.run(debug=True)