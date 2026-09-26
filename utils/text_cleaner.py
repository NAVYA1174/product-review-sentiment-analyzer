"""
Text preprocessing and cleaning utilities for Product Review Sentiment Analysis.
Handles contractions, emojis, negation tagging, tokenization, and sentiment keywords.
"""

import re
import string

# Common English contractions dictionary
CONTRACTIONS = {
    "didn't": "did not",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "it's": "it is",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "they'll": "they will",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "let's": "let us",
}

# Negation words that flip polarity
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "scarcely",
    "barely", "doesn't", "don't", "didn't", "wasn't", "weren't", "isn't",
    "aren't", "cannot", "couldn't", "wouldn't", "shouldn't", "won't"
}

# Common emoji translations for product sentiment
EMOJI_SENTIMENT = {
    "❤️": " love ",
    "😍": " wonderful love ",
    "👍": " thumbs up great ",
    "👎": " thumbs down terrible ",
    "⭐": " star five ",
    "🔥": " awesome fire ",
    "🎉": " great celebration ",
    "😊": " happy good ",
    "😃": " very happy excellent ",
    "🙂": " pleased good ",
    "😡": " angry terrible furious ",
    "😠": " upset mad ",
    "🤬": " very angry horrific ",
    "😭": " crying devastated ",
    "😢": " sad disappointed ",
    "💔": " broken heart defective ",
    "🤮": " disgusted awful ",
    "💩": " rubbish junk terrible ",
    "💯": " perfect excellent ",
    "✨": " brilliant top tier ",
    "👌": " perfect satisfactory ",
}

def expand_contractions(text: str) -> str:
    """Expands contractions like don't -> do not."""
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text_lower)
    return text_lower

def replace_emojis(text: str) -> str:
    """Replaces common emojis with their sentiment-equivalent text."""
    for emoji, sentiment_word in EMOJI_SENTIMENT.items():
        if emoji in text:
            text = text.replace(emoji, sentiment_word)
    return text

def clean_text(text: str, remove_stopwords: bool = False, keep_negations: bool = True) -> str:
    """
    Cleans raw review text for ML and sentiment analysis.
    - Handles HTML tags and URLs
    - Translates sentiment emojis
    - Expands contractions
    - Optionally removes non-negation stopwords
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Replace emojis with sentiment descriptors
    cleaned = replace_emojis(text)

    # Remove HTML tags & links
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = re.sub(r'http[s]?://\S+', ' ', cleaned)

    # Expand contractions
    cleaned = expand_contractions(cleaned)

    # Normalize multiple punctuation (e.g., '!!!!' -> '!', '??' -> '?')
    cleaned = re.sub(r'([!?.]){2,}', r'\1', cleaned)

    # Clean whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if remove_stopwords:
        tokens = cleaned.split()
        default_stops = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "during", "each", "for", "from", "further",
            "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
            "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
            "me", "more", "most", "my", "myself", "of", "off", "on", "once", "only", "or",
            "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
            "she", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
            "themselves", "then", "there", "these", "they", "this", "those", "through",
            "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
            "when", "where", "which", "while", "who", "whom", "why", "with", "you", "your",
            "yours", "yourself", "yourselves"
        }
        if keep_negations:
            stop_set = default_stops - NEGATION_WORDS
        else:
            stop_set = default_stops

        tokens = [t for t in tokens if t.lower() not in stop_set]
        cleaned = " ".join(tokens)

    return cleaned

def extract_sentiment_keywords(text: str) -> dict:
    """
    Extracts explicitly positive and negative tokens/phrases from a review.
    Useful for highlighting in UI and generating word clouds.
    """
    cleaned = clean_text(text).lower()
    words = re.findall(r'\b[a-z]{3,}\b', cleaned)

    pos_lexicon = {
        "great", "excellent", "amazing", "love", "awesome", "fantastic", "perfect",
        "good", "best", "impressive", "sturdy", "durable", "fast", "smooth", "flawless",
        "recommend", "superb", "brilliant", "pleased", "worth", "satisfied", "comfortable",
        "crisp", "clear", "premium", "top", "nice", "reliable", "beautiful", "solid",
        "bargain", "easy", "outstanding", "exceptional", "delight", "favorite", "genius"
    }

    neg_lexicon = {
        "bad", "terrible", "horrible", "awful", "poor", "worst", "hate", "waste",
        "broken", "broke", "defective", "disappointed", "disappointing", "slow", "useless",
        "cheap", "flimsy", "garbage", "trash", "return", "refund", "annoying", "fails",
        "failed", "stopped", "noisy", "uncomfortable", "scam", "overpriced", "laggy",
        "painful", "defect", "faulty", "regret", "problem", "issues", "error", "cracked"
    }

    pos_found = [w for w in words if w in pos_lexicon]
    neg_found = [w for w in words if w in neg_lexicon]

    return {
        "positive": list(set(pos_found)),
        "negative": list(set(neg_found)),
        "pos_count": len(pos_found),
        "neg_count": len(neg_found)
    }
