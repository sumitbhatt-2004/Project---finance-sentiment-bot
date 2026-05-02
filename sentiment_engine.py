from textblob import TextBlob

def get_sentiment_score(text):
    """
    Analyze the sentiment of a news headline.
    Returns a score between -1 (Negative) and 1 (Positive).
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Business Logic:
# If Sentiment > 0 and LSTM predicts UP -> Strong Buy
# If Sentiment < 0 and LSTM predicts UP -> Potential Trap/High Risk
