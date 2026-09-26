"""
Aspect-Based Sentiment Analysis (ABSA) module.
Identifies product dimensions (Build Quality, Performance, Price/Value, Design/Comfort, Support)
and calculates aspect-level sentiment scores with extracted evidence quotes.
"""

import re
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

ASPECT_TAXONOMY = {
    "Quality & Build": [
        "quality", "build", "material", "durability", "durable", "sturdy", "solid",
        "flimsy", "broke", "broken", "cheaply made", "hinge", "casing", "crack",
        "cracked", "peeling", "craftsmanship", "shatter", "scratch", "leak", "leaks"
    ],
    "Performance": [
        "performance", "battery", "sound", "audio", "bass", "treble", "speed",
        "fast", "slow", "noise cancellation", "anc", "mic", "microphone", "display",
        "screen", "lag", "glitch", "sensor", "ecg", "heart rate", "steam", "heat",
        "cook", "cooks", "crispy", "suction", "charge", "charging", "processor"
    ],
    "Price & Value": [
        "price", "value", "worth", "cost", "money", "affordable", "bargain",
        "expensive", "overpriced", "cheap", "penny", "rip-off", "investment"
    ],
    "Design & Comfort": [
        "design", "comfort", "comfortable", "ergonomic", "fit", "fits", "slip",
        "weight", "lightweight", "heavy", "bulky", "sleek", "look", "looks",
        "style", "strap", "uncomfortable", "blister", "blisters", "size", "cushion"
    ],
    "Support & Delivery": [
        "service", "customer service", "support", "warranty", "return", "refund",
        "shipping", "delivery", "arrived", "packaging", "box", "package", "exchange"
    ]
}

class AspectExtractor:
    def __init__(self):
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception:
            self.vader = None

    def _split_into_sentences(self, text: str) -> list:
        """Splits review text into clean sentence clauses."""
        # Split on sentence boundaries (. ! ? \n ;)
        sentences = re.split(r'[.!?;\n]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def _score_sentence(self, sentence: str) -> float:
        """Scores a single sentence sentiment with domain-specific modifiers."""
        if self.vader:
            v_score = self.vader.polarity_scores(sentence)["compound"]
        else:
            v_score = 0.0

        tb_score = TextBlob(sentence).sentiment.polarity
        base_score = 0.6 * v_score + 0.4 * tb_score

        # Domain-specific word adjustments for product aspects
        sent_lower = sentence.lower()
        neg_clues = ["flimsy", "broke", "broken", "cheaply", "cheap", "crack", "cracked", "leaks", "creak", "shoddy", "defective", "uncomfortable", "blister", "lag", "painful", "stiff", "scratch"]
        pos_clues = ["sturdy", "solid", "durable", "premium", "comfortable", "plush", "crisp", "clear", "flawless", "smooth", "punchy", "fast", "reliable"]

        for nc in neg_clues:
            if re.search(r'\b' + re.escape(nc) + r'\b', sent_lower):
                base_score -= 0.35

        for pc in pos_clues:
            if re.search(r'\b' + re.escape(pc) + r'\b', sent_lower):
                base_score += 0.25

        # Clamp between -1.0 and 1.0
        return round(max(-1.0, min(1.0, base_score)), 3)

    def extract_aspects(self, text: str) -> dict:
        """
        Extracts mentions and sentiment for all taxonomy aspects.
        Returns detailed aspect cards with score, sentiment label, and quotes.
        """
        sentences = self._split_into_sentences(text)
        results = {}

        for aspect_name, keywords in ASPECT_TAXONOMY.items():
            matched_sentences = []
            matched_keywords = []

            for sent in sentences:
                sent_lower = sent.lower()
                for kw in keywords:
                    # Match whole word or phrase
                    if re.search(r'\b' + re.escape(kw) + r'\b', sent_lower):
                        matched_sentences.append(sent)
                        matched_keywords.append(kw)
                        break

            if matched_sentences:
                # Calculate aggregate score across all sentences mentioning this aspect
                scores = [self._score_sentence(s) for s in matched_sentences]
                avg_score = round(sum(scores) / len(scores), 3)

                if avg_score >= 0.08:
                    sentiment = "Positive"
                elif avg_score <= -0.08:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                results[aspect_name] = {
                    "mentioned": True,
                    "score": avg_score,
                    "sentiment": sentiment,
                    "mentions_count": len(matched_sentences),
                    "matched_keywords": list(set(matched_keywords)),
                    "evidence_quotes": matched_sentences
                }
            else:
                results[aspect_name] = {
                    "mentioned": False,
                    "score": 0.0,
                    "sentiment": "Not Mentioned",
                    "mentions_count": 0,
                    "matched_keywords": [],
                    "evidence_quotes": []
                }

        return results

    def batch_extract_aspects(self, review_texts: list) -> dict:
        """
        Aggregates aspect sentiment statistics across a batch of reviews.
        Useful for product summary dashboard and radar charts.
        """
        summary = {
            aspect: {"positive": 0, "neutral": 0, "negative": 0, "total_mentions": 0, "scores": []}
            for aspect in ASPECT_TAXONOMY
        }

        for text in review_texts:
            aspect_res = self.extract_aspects(text)
            for aspect, data in aspect_res.items():
                if data["mentioned"]:
                    summary[aspect]["total_mentions"] += 1
                    summary[aspect]["scores"].append(data["score"])
                    sent_key = data["sentiment"].lower()
                    if sent_key in summary[aspect]:
                        summary[aspect][sent_key] += 1

        # Calculate averages and Net Sentiment Score (NSS = %Pos - %Neg)
        processed_summary = {}
        for aspect, stats in summary.items():
            tot = stats["total_mentions"]
            if tot > 0:
                avg_score = round(sum(stats["scores"]) / tot, 3)
                pos_pct = round((stats["positive"] / tot) * 100, 1)
                neg_pct = round((stats["negative"] / tot) * 100, 1)
                net_sentiment = round(pos_pct - neg_pct, 1)
            else:
                avg_score = 0.0
                pos_pct = 0.0
                neg_pct = 0.0
                net_sentiment = 0.0

            processed_summary[aspect] = {
                "total_mentions": tot,
                "average_score": avg_score,
                "positive_count": stats["positive"],
                "neutral_count": stats["neutral"],
                "negative_count": stats["negative"],
                "positive_pct": pos_pct,
                "negative_pct": neg_pct,
                "net_sentiment_score": net_sentiment
            }

        return processed_summary
