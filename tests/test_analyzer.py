"""
Unit tests for Product Review Sentiment Analyzer pipeline.
Tests text cleaning, sentiment engines, ABSA, and report generation.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.text_cleaner import clean_text, expand_contractions, extract_sentiment_keywords
from models.sentiment_engine import SentimentEngine
from models.aspect_extractor import AspectExtractor

class TestSentimentPipeline(unittest.TestCase):
    def setUp(self):
        model_path = os.path.join(BASE_DIR, "models", "saved_models", "sentiment_classifier.joblib")
        vec_path = os.path.join(BASE_DIR, "models", "saved_models", "tfidf_vectorizer.joblib")
        self.engine = SentimentEngine(ml_model_path=model_path, vectorizer_path=vec_path)
        self.aspect_extractor = AspectExtractor()

    def test_text_cleaner(self):
        text = "I didn't like the product! It's super broken ❤️"
        cleaned = clean_text(text)
        self.assertIn("not", cleaned)
        self.assertIn("love", cleaned)

    def test_sentiment_positive(self):
        review = "This wireless earbud has phenomenal sound quality and amazing battery life! Absolutely love it."
        res = self.engine.analyze(review)
        self.assertEqual(res["label"], "Positive")
        self.assertGreater(res["score"], 0.2)
        self.assertGreaterEqual(res["estimated_stars"], 4.0)

    def test_sentiment_negative(self):
        review = "Horrible waste of money. The hinge broke on day two and customer service was unhelpful and rude."
        res = self.engine.analyze(review)
        self.assertEqual(res["label"], "Negative")
        self.assertLess(res["score"], -0.2)
        self.assertLessEqual(res["estimated_stars"], 2.5)

    def test_sarcasm_detection(self):
        sarcastic_review = "Yeah right, supreme noise cancellation! The only noise it cancelled was my expectation. Broke in 2 days."
        res = self.engine.analyze(sarcastic_review)
        self.assertTrue(res["is_sarcastic"])
        self.assertEqual(res["label"], "Negative")

    def test_aspect_extraction(self):
        text = "Battery life easily lasted 8 hours. However, the build quality feels flimsy and cheap."
        aspects = self.aspect_extractor.extract_aspects(text)
        self.assertTrue(aspects["Performance"]["mentioned"])
        self.assertTrue(aspects["Quality & Build"]["mentioned"])
        self.assertEqual(aspects["Quality & Build"]["sentiment"], "Negative")

if __name__ == "__main__":
    unittest.main()
