"""
Core Multi-Engine Sentiment Analysis Module.
Supports VADER, TextBlob, trained ML classifiers, and Ensemble scoring.
Includes emotion detection, star rating estimation, and sarcasm heuristics.
"""

import os
import re
import joblib
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils.text_cleaner import clean_text, extract_sentiment_keywords

class SentimentEngine:
    def __init__(self, ml_model_path: str = None, vectorizer_path: str = None):
        # Initialize VADER
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception:
            self.vader = None

        # Load ML model if present
        self.ml_model = None
        self.vectorizer = None
        if ml_model_path and os.path.exists(ml_model_path) and vectorizer_path and os.path.exists(vectorizer_path):
            try:
                self.ml_model = joblib.load(ml_model_path)
                self.vectorizer = joblib.load(vectorizer_path)
            except Exception as e:
                print(f"Warning: Could not load ML model: {e}")

    def analyze_vader(self, text: str) -> dict:
        """Runs VADER sentiment intensity analysis."""
        if not self.vader or not text.strip():
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "Neutral"}
        
        scores = self.vader.polarity_scores(text)
        comp = scores["compound"]
        if comp >= 0.05:
            label = "Positive"
        elif comp <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "compound": float(comp),
            "pos": float(scores["pos"]),
            "neu": float(scores["neu"]),
            "neg": float(scores["neg"]),
            "label": label
        }

    def analyze_textblob(self, text: str) -> dict:
        """Runs TextBlob sentiment polarity and subjectivity analysis."""
        if not text.strip():
            return {"polarity": 0.0, "subjectivity": 0.0, "label": "Neutral"}

        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)

        if polarity > 0.05:
            label = "Positive"
        elif polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "label": label
        }

    def analyze_ml(self, text: str) -> dict:
        """Runs trained scikit-learn ML model prediction."""
        if not self.ml_model or not self.vectorizer:
            return None

        cleaned = clean_text(text, remove_stopwords=False)
        vec = self.vectorizer.transform([cleaned])
        pred_label = self.ml_model.predict(vec)[0]

        # Probabilities
        if hasattr(self.ml_model, "predict_proba"):
            probs = self.ml_model.predict_proba(vec)[0]
            classes = list(self.ml_model.classes_)
            prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
            confidence = float(np.max(probs))
        else:
            prob_dict = {}
            confidence = 1.0

        return {
            "label": str(pred_label),
            "confidence": confidence,
            "probabilities": prob_dict
        }

    def detect_sarcasm(self, text: str) -> bool:
        """
        Rule-based heuristics for sarcastic / contradictory reviews.
        Looks for patterns like 'yeah right', quotes around superlative words, or contrasting clauses.
        """
        text_lower = text.lower()
        sarcasm_triggers = [
            r"yeah right",
            r"oh wonderful.*(broken|terrible|defective|useless)",
            r"great.*if you enjoy.*(noise|pain|static|losing)",
            r"best.*(waste|mistake|joke)",
            r"\"amazing\".*(broke|failed|never)",
            r"\"perfect\".*(garbage|scam)",
            r"five stars for.*zero stars for"
        ]
        for pattern in sarcasm_triggers:
            if re.search(pattern, text_lower):
                return True
        return False

    def detect_emotions(self, text: str) -> dict:
        """Detects emotional tones in review: Joy, Trust, Frustration, Disappointment."""
        text_lower = text.lower()

        joy_words = ["love", "amazing", "awesome", "fantastic", "superb", "delight", "pleased", "great", "wonderful"]
        trust_words = ["reliable", "solid", "sturdy", "recommend", "accurate", "durable", "consistent", "flawless"]
        frustration_words = ["horrible", "terrible", "waste", "garbage", "trash", "hate", "scam", "annoying", "furious"]
        disappointment_words = ["disappointed", "regret", "stopped", "broke", "defective", "cracked", "flimsy", "failed"]

        def count_matches(words):
            return sum(1 for w in words if re.search(r'\b' + re.escape(w) + r'\b', text_lower))

        j_score = count_matches(joy_words)
        t_score = count_matches(trust_words)
        f_score = count_matches(frustration_words)
        d_score = count_matches(disappointment_words)

        total = j_score + t_score + f_score + d_score
        if total == 0:
            return {"Joy": 25, "Trust": 25, "Frustration": 25, "Disappointment": 25}

        return {
            "Joy": round((j_score / total) * 100, 1),
            "Trust": round((t_score / total) * 100, 1),
            "Frustration": round((f_score / total) * 100, 1),
            "Disappointment": round((d_score / total) * 100, 1)
        }

    def estimate_star_rating(self, compound_score: float) -> float:
        """
        Converts a -1.0 to +1.0 sentiment score into estimated 1 to 5 star rating.
        """
        # Map [-1, 1] to [1, 5] linearly: rating = 3 + 2 * score
        rating = 3.0 + (compound_score * 2.0)
        # Round to nearest 0.5
        rating = round(rating * 2) / 2
        return max(1.0, min(5.0, rating))

    def analyze(self, text: str, engine: str = "ensemble") -> dict:
        """
        Comprehensive review analysis returning sentiment, polarity, emotions,
        keywords, sarcasm flags, and estimated stars.
        Engine choices: 'vader', 'textblob', 'ml', 'ensemble'
        """
        if not text or not text.strip():
            return {
                "text": text,
                "label": "Neutral",
                "score": 0.0,
                "confidence": 0.5,
                "engine_used": engine,
                "estimated_stars": 3.0,
                "is_sarcastic": False,
                "emotions": {"Joy": 25, "Trust": 25, "Frustration": 25, "Disappointment": 25},
                "keywords": {"positive": [], "negative": []}
            }

        vader_res = self.analyze_vader(text)
        tb_res = self.analyze_textblob(text)
        ml_res = self.analyze_ml(text)
        is_sarcastic = self.detect_sarcasm(text)

        # Compute consensus score depending on engine chosen
        if engine == "vader":
            score = vader_res["compound"]
            label = vader_res["label"]
            confidence = abs(score)
        elif engine == "textblob":
            score = tb_res["polarity"]
            label = tb_res["label"]
            confidence = abs(score)
        elif engine == "ml" and ml_res:
            label = ml_res["label"]
            score = 0.8 if label == "Positive" else (-0.8 if label == "Negative" else 0.0)
            confidence = ml_res["confidence"]
        else:
            # Ensemble weighted consensus
            # 55% VADER, 45% TextBlob
            score = (0.55 * vader_res["compound"]) + (0.45 * tb_res["polarity"])
            if ml_res:
                ml_val = 0.8 if ml_res["label"] == "Positive" else (-0.8 if ml_res["label"] == "Negative" else 0.0)
                score = (0.4 * vader_res["compound"]) + (0.3 * tb_res["polarity"]) + (0.3 * ml_val)

            # Sarcasm override if strong sarcastic pattern found
            if is_sarcastic and score > 0:
                score = -0.65
                label = "Negative"
            else:
                if score >= 0.05:
                    label = "Positive"
                elif score <= -0.05:
                    label = "Negative"
                else:
                    label = "Neutral"

            confidence = min(1.0, max(0.4, abs(score) + 0.2))

        estimated_stars = self.estimate_star_rating(score)
        emotions = self.detect_emotions(text)
        keywords = extract_sentiment_keywords(text)

        return {
            "text": text,
            "label": label,
            "score": round(score, 4),
            "confidence": round(confidence, 3),
            "engine_used": engine,
            "vader_scores": vader_res,
            "textblob_scores": tb_res,
            "ml_scores": ml_res,
            "estimated_stars": estimated_stars,
            "is_sarcastic": is_sarcastic,
            "emotions": emotions,
            "keywords": keywords
        }
