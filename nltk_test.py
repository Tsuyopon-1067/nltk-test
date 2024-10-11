import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# NLTKのリソースをダウンロード（初回のみ必要）
nltk.download('vader_lexicon')


def extract_emotion_nltk(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def extract_emotion_transformers(text):
    classifier = pipeline(
        "text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    result = classifier(text)
    emotions = result[0]
    return max(emotions, key=lambda x: x['score'])['label']


# テスト文
test_sentences = [
    "I'm so happy today!",
    "This situation makes me very angry and frustrated.",
    "I'm feeling a bit down and sad.",
    "Wow, I'm really surprised by this news!",
    "I'm just going about my day as usual."
]

print("NLTK Sentiment Analysis:")
for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Emotion: {extract_emotion_nltk(sentence)}\n")

print("\nTransformers Emotion Classification:")
for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Emotion: {extract_emotion_transformers(sentence)}\n")
