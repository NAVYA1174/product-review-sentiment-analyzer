"""
Training and evaluation script for Product Review Sentiment Classifier.
Trains TF-IDF + Logistic Regression / Naive Bayes on product reviews.
Saves serialized model and vectorizer for production inference.
"""

import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from utils.text_cleaner import clean_text

def train_and_save_model(data_path: str = None, output_dir: str = None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "sample_reviews.csv")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "models", "saved_models")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Map star rating to 3-class sentiment: Positive, Neutral, Negative
    def rating_to_sentiment(r):
        if r >= 4:
            return "Positive"
        elif r <= 2:
            return "Negative"
        else:
            return "Neutral"

    if "sentiment" not in df.columns and "star_rating" in df.columns:
        df["sentiment"] = df["star_rating"].apply(rating_to_sentiment)

    # Clean review text
    df["cleaned_text"] = df["review_text"].apply(lambda t: clean_text(str(t), remove_stopwords=False))

    # Features and labels
    X = df["cleaned_text"]
    y = df["sentiment"]

    print(f"Class distribution:\n{y.value_counts()}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer with unigrams & bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=1
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model 1: Logistic Regression
    lr_model = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    lr_f1 = f1_score(y_test, lr_preds, average="weighted")
    lr_acc = accuracy_score(y_test, lr_preds)

    # Model 2: Multinomial Naive Bayes
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    nb_f1 = f1_score(y_test, nb_preds, average="weighted")
    nb_acc = accuracy_score(y_test, nb_preds)

    print(f"Logistic Regression - Acc: {lr_acc:.3f}, F1: {lr_f1:.3f}")
    print(f"Multinomial NB      - Acc: {nb_acc:.3f}, F1: {nb_f1:.3f}")

    # Select best model
    if lr_f1 >= nb_f1:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_preds = lr_preds
    else:
        best_model = nb_model
        best_name = "Multinomial Naive Bayes"
        best_preds = nb_preds

    print(f"\nSelected Best Model: {best_name}")
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds))

    # Save artifacts
    model_path = os.path.join(output_dir, "sentiment_classifier.joblib")
    vec_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vec_path}")

    return {
        "best_model_name": best_name,
        "accuracy": float(accuracy_score(y_test, best_preds)),
        "f1_score": float(f1_score(y_test, best_preds, average="weighted")),
        "classification_report": classification_report(y_test, best_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, best_preds, labels=["Positive", "Neutral", "Negative"]).tolist()
    }

if __name__ == "__main__":
    train_and_save_model()
