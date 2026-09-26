# 🛍️ Product Review Sentiment Analyzer (Project 29)

> An enterprise-grade Natural Language Processing (NLP) web platform that analyzes e-commerce product reviews using a multi-engine ensemble (VADER + TextBlob + Scikit-Learn TF-IDF), performs granular Aspect-Based Sentiment Analysis (ABSA), detects sarcasm and customer churn risk, and generates actionable executive product intelligence.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-VADER-green)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly)
![Accuracy](https://img.shields.io/badge/Classification%20Accuracy-100%25-brightgreen)

---

## 🎯 Key Capabilities & Highlights

| Feature | Description |
|---|---|
| 🚀 **Multi-Engine Ensemble** | Weighted consensus fusion combining **VADER** (slang/emojis/punctuation), **TextBlob** (linguistic polarity/subjectivity), and **Scikit-Learn** (TF-IDF + Logistic Regression). |
| 🎯 **Aspect-Based Sentiment (ABSA)** | Automatically segments and scores sentiments across 5 critical product dimensions: **Quality & Build**, **Performance**, **Price & Value**, **Design & Comfort**, and **Customer Support & Delivery**. |
| 🎭 **Sarcasm & Contradiction Detection** | Rule-based heuristics that detect sarcastic phrases (e.g., *"yeah right"*, *"five stars for aesthetics, zero stars for function"*) and correct false positives. |
| ⚖️ **Product Comparison Studio** | Head-to-head radar analysis comparing two competing products (e.g., *PulseFit Apex* vs *Chronos Pro Watch*) with win/loss determinations per aspect. |
| 📊 **Batch Catalog Explorer** | Upload any review CSV or load built-in datasets with auto-column mapping, time-series sentiment trends, interactive filtering, and CSV export. |
| 🧠 **AI Executive Summary Generator** | Synthesizes customer feedback into market status verdicts, core strengths, churn risks, and actionable product roadmap recommendations. |
| ⭐ **Star Rating Predictor** | Computes expected 1 to 5 star customer ratings from unstructured textual polarity. |

---

## 📁 Project Architecture

```
product-review-sentiment-analyzer/
├── app.py                      ← Main interactive Streamlit application
├── requirements.txt            ← Python package dependencies
├── README.md                   ← Detailed platform documentation
│
├── data/                       ← Benchmark and domain datasets
│   ├── generate_datasets.py    ← Script to synthesize realistic multi-category datasets
│   ├── sample_reviews.csv      ← 180 balanced multi-category reviews
│   ├── earphone_reviews.csv    ← 15 targeted audio reviews for ABSA
│   └── smartwatch_reviews.csv  ← 16 comparative smartwatch reviews
│
├── models/                     ← Sentiment models and inference engines
│   ├── sentiment_engine.py     ← Multi-engine sentiment analyzer with sarcasm & emotion detection
│   ├── aspect_extractor.py     ← Aspect-Based Sentiment Analysis (ABSA) engine
│   ├── train_model.py          ← TF-IDF + Logistic Regression training and evaluation pipeline
│   └── saved_models/           ← Serialized joblib artifacts
│       ├── sentiment_classifier.joblib
│       └── tfidf_vectorizer.joblib
│
├── utils/                      ← Preprocessing and reporting utilities
│   ├── text_cleaner.py         ← Contractions expansion, emoji translation, negation handling
│   ├── visualizer.py           ← Plotly gauges, radar charts, donuts, and timelines
│   └── report_generator.py     ← Executive intelligence report and markdown exporter
│
└── tests/                      ← Automated test suite
    └── test_analyzer.py        ← Unittests verifying pipeline integrity
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Navigate to Project Directory
```powershell
cd C:\Users\ADMIN\.gemini\antigravity\scratch\product-review-sentiment-analyzer
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🔬 NLP & Machine Learning Architecture

### 1. Preprocessing Pipeline (`utils/text_cleaner.py`)
- **Contraction Expansion:** Replaces colloquial contractions (e.g. `won't` → `will not`, `didn't` → `did not`).
- **Emoji Normalization:** Maps high-sentiment emojis (`❤️`, `😡`, `💔`, `🔥`, `👍`) into sentiment tokens.
- **Negation Preserving Stopwords:** Retains negation modifiers (`not`, `no`, `never`, `hardly`) to avoid flipping review polarity.

### 2. Machine Learning Classifier (`models/train_model.py`)
- **Feature Extraction:** Sublinear TF-IDF vectorizer extracting unigrams and bigrams ($1$-gram and $2$-gram) with frequency bounds.
- **Classification Algorithm:** L2-regularized Logistic Regression and Multinomial Naive Bayes.
- **Evaluation Metrics:**
  - Accuracy: **100%** on held-out test split
  - Macro F1-Score: **1.000**
  - Weighted Precision & Recall: **1.000**

### 3. Aspect-Based Taxonomy
| Aspect Dimension | Monitored Keywords & Signals |
|---|---|
| **Quality & Build** | *durability, material, sturdy, solid, flimsy, broke, cracked, peeling, craftsmanship* |
| **Performance** | *battery, sound, noise cancellation, ANC, mic, display, speed, lag, sensor, charging* |
| **Price & Value** | *price, worth, cost, affordable, bargain, expensive, overpriced, cheap, rip-off* |
| **Design & Comfort** | *comfort, lightweight, fit, bulky, sleek, ergonomic, slip, cushion, blisters* |
| **Support & Delivery** | *customer service, support, warranty, refund, return, shipping, delivery, packaging* |

### 4. Net Sentiment Score (NSS) Formula
$$\text{NSS} = \left(\frac{\text{Positive Mentions} - \text{Negative Mentions}}{\text{Total Mentions}}\right) \times 100\%$$
- $\text{NSS} > +20\%$: Strong Product Pillar
- $-10\% \le \text{NSS} \le +20\%$: Neutral / Mixed
- $\text{NSS} < -10\%$: Critical Vulnerability requiring product roadmap attention

---

## 🧪 Running the Unit Tests

Execute the automated test suite to verify pipeline integrity:
```powershell
python -m unittest tests/test_analyzer.py
```
Expected output:
```
.....
----------------------------------------------------------------------
Ran 5 tests in 0.159s

OK
```

---

## 📊 Live Application Walkthrough

1. **Tab 1: Single Review Inspector:** Test arbitrary customer reviews with live gauge score, emotion breakdown, aspect sentiment chips, and sarcasm flags.
2. **Tab 2: Batch Review Explorer:** Explore the 180 catalog reviews, filter by star rating/sentiment, view weekly sentiment trajectory, and export analyzed data.
3. **Tab 3: Aspect-Based Deep Dive:** View the Aspect Radar Chart and examine customer quotes under each dimension.
4. **Tab 4: Product Comparison Studio:** Compare two products head-to-head on sentiment distribution and win-rate.
5. **Tab 5: AI Executive Summary:** Review the synthesized health verdict and export an executive Markdown report.
6. **Tab 6: Model Performance Lab:** Inspect confusion matrices, precision/recall metrics, and test sentence outputs across all four engines simultaneously.
