from textblob import TextBlob

sentences = [
    "I love this new technology",
    "This is a terrible accident",
    "The match was okay"
]

for s in sentences:
    blob = TextBlob(s)
    print(s, "->", blob.sentiment.polarity)
