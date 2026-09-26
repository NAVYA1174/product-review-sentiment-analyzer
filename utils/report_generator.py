"""
Report and Executive Summary Generator.
Synthesizes batch sentiment metrics, aspect strengths and weaknesses,
and generates actionable product intelligence reports.
"""

import pandas as pd
from datetime import datetime

def generate_executive_insights(df: pd.DataFrame, aspect_summary: dict) -> dict:
    """
    Generates structured AI-driven executive insights, pros, cons, and recommendations.
    """
    total_reviews = len(df)
    if total_reviews == 0:
        return {}

    # Sentiment distribution
    sent_counts = df["sentiment_label"].value_counts().to_dict()
    pos_cnt = sent_counts.get("Positive", 0)
    neg_cnt = sent_counts.get("Negative", 0)
    neu_cnt = sent_counts.get("Neutral", 0)

    pos_pct = round((pos_cnt / total_reviews) * 100, 1)
    neg_pct = round((neg_cnt / total_reviews) * 100, 1)
    avg_score = round(df["sentiment_score"].mean(), 2)

    # Average star rating if available
    avg_stars = round(df["star_rating"].mean(), 1) if "star_rating" in df.columns else None

    # Identify Top Performing Aspects (Highest Net Sentiment)
    sorted_aspects = sorted(
        aspect_summary.items(),
        key=lambda item: item[1]["net_sentiment_score"],
        reverse=True
    )

    top_strengths = [
        f"{asp} (+{data['net_sentiment_score']}% Net Sentiment, {data['total_mentions']} mentions)"
        for asp, data in sorted_aspects if data["net_sentiment_score"] > 0 and data["total_mentions"] > 0
    ][:3]

    top_weaknesses = [
        f"{asp} ({data['net_sentiment_score']}% Net Sentiment, {data['total_mentions']} mentions)"
        for asp, data in sorted(sorted_aspects, key=lambda x: x[1]["net_sentiment_score"])
        if data["net_sentiment_score"] < 0 and data["total_mentions"] > 0
    ][:3]

    # Generate Actionable Recommendations for Product Managers
    recommendations = []
    for asp, data in sorted_aspects:
        if data["net_sentiment_score"] < -10 and data["total_mentions"] > 0:
            if "Quality" in asp:
                recommendations.append("Strengthen component durability QA testing and supplier quality control on housing/hinges.")
            elif "Performance" in asp:
                recommendations.append("Investigate thermal management, battery drain, or firmware optimizations to address performance lag.")
            elif "Support" in asp:
                recommendations.append("Streamline customer support response SLAs and simplify the replacement/warranty return process.")
            elif "Design" in asp:
                recommendations.append("Re-evaluate ergonomic dimensions and include alternate tip/strap sizes in packaging.")
            elif "Price" in asp:
                recommendations.append("Adjust price positioning or bundle promotional value-add accessories.")

    if not recommendations:
        recommendations.append("Maintain current product standards while expanding marketing on high-satisfaction features.")

    # High-level executive verdict
    if pos_pct >= 70:
        verdict = "Market Leader — High Customer Loyalty & Organic Advocacy"
        verdict_badge = "Excellent"
    elif pos_pct >= 50:
        verdict = "Competitive Contender — Healthy Demand with Addressable Pain Points"
        verdict_badge = "Good"
    elif neg_pct >= 50:
        verdict = "At-Risk Product — Critical Churn & Quality Red Flags Detected"
        verdict_badge = "Critical"
    else:
        verdict = "Mixed Market Perception — Inconsistent Customer Experience"
        verdict_badge = "Moderate"

    return {
        "total_reviews": total_reviews,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct": round(100 - pos_pct - neg_pct, 1),
        "average_sentiment_score": avg_score,
        "average_stars": avg_stars,
        "verdict": verdict,
        "verdict_badge": verdict_badge,
        "top_strengths": top_strengths if top_strengths else ["General product satisfaction across baseline metrics"],
        "top_weaknesses": top_weaknesses if top_weaknesses else ["No acute aspect vulnerabilities identified"],
        "recommendations": recommendations
    }

def generate_markdown_report(product_name: str, insights: dict) -> str:
    """Formats executive insights into a clean exportable Markdown report."""
    now = datetime.now().strftime("%B %d, %Y - %H:%M")
    md = f"""# 📊 Product Review Sentiment Intelligence Report
**Product:** {product_name}  
**Generated At:** {now}  
**Total Reviews Analyzed:** {insights.get('total_reviews', 0)}  

---

### 🏆 Executive Summary
- **Market Status:** {insights.get('verdict', 'N/A')}
- **Positive Sentiment Ratio:** {insights.get('positive_pct', 0)}%
- **Negative Sentiment Ratio:** {insights.get('negative_pct', 0)}%
- **Average Sentiment Score:** {insights.get('average_sentiment_score', 0):+.2f} / 1.00
"""
    if insights.get("average_stars"):
        md += f"- **Average Star Rating:** {insights['average_stars']} / 5.0 ⭐\n"

    md += "\n### 🌟 Core Product Strengths\n"
    for s in insights.get("top_strengths", []):
        md += f"- {s}\n"

    md += "\n### ⚠️ Critical Pain Points\n"
    for w in insights.get("top_weaknesses", []):
        md += f"- {w}\n"

    md += "\n### 🎯 Actionable Product Recommendations\n"
    for r in insights.get("recommendations", []):
        md += f"1. {r}\n"

    md += "\n---\n*Report generated by Product Review Sentiment Analyzer (Project 29)*\n"
    return md
