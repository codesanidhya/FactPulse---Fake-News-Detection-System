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
