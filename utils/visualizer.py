"""
Interactive Visualization Utilities using Plotly and WordCloud.
Produces gauges, radar charts, distribution donuts, sentiment timelines,
aspect comparisons, and word clouds.
"""

import io
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Consistent Theme Color Palette
COLORS = {
    "Positive": "#10B981",    # Emerald
    "Neutral": "#F59E0B",     # Amber
    "Negative": "#EF4444",    # Rose / Red
    "Primary": "#3B82F6",     # Blue
    "Secondary": "#8B5CF6",   # Purple
    "Background": "#0F172A",  # Dark Slate
    "CardBg": "#1E293B",      # Slate 800
    "Text": "#F8FAFC"         # Light
}

def create_sentiment_gauge(score: float, title: str = "Sentiment Polarity") -> go.Figure:
    """
    Renders an animated dial gauge for sentiment score between -1.0 and +1.0.
    """
    # Scale -1.0 .. +1.0 to 0 .. 100 for gauge rendering
    val_100 = (score + 1.0) * 50

    if score > 0.05:
        bar_color = "#10B981"
    elif score < -0.05:
        bar_color = "#EF4444"
    else:
        bar_color = "#F59E0B"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 2),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b><br><span style='font-size:0.8em;color:#94A3B8'>Range: -1.0 (Very Neg) to +1.0 (Very Pos)</span>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [-1.0, 1.0], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(30, 41, 59, 0.6)",
            'borderwidth': 1,
            'bordercolor': "#475569",
            'steps': [
                {'range': [-1.0, -0.05], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [-0.05, 0.05], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [0.05, 1.0], 'color': "rgba(16, 185, 129, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.8,
                'value': round(score, 2)
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#F8FAFC", 'family': "Inter, sans-serif"},
        height=260,
        margin=dict(l=25, r=25, t=50, b=20)
    )
    return fig

def create_sentiment_donut(counts: dict) -> go.Figure:
    """
    Renders sleek Donut chart showing Positive / Neutral / Negative proportions.
    """
    labels = list(counts.keys())
    values = list(counts.values())
    color_map = [COLORS.get(label, "#94A3B8") for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=color_map, line=dict(color='#0F172A', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        insidetextorientation='radial'
    )])

    total = sum(values)
    pos_count = counts.get("Positive", 0)
    pos_ratio = f"{round((pos_count / total)*100)}%" if total > 0 else "0%"

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#CBD5E1")),
        annotations=[dict(text=f"<b>{pos_ratio}</b><br><span style='font-size:11px;color:#94A3B8'>Positive</span>",
                          x=0.5, y=0.5, font_size=20, font_color="#10B981", showarrow=False)],
        height=290,
        margin=dict(l=10, r=10, t=20, b=40)
    )
    return fig

def create_aspect_radar(aspect_scores: dict) -> go.Figure:
    """
    Renders Radar (Spider) chart of scores across product dimensions.
    """
    categories = list(aspect_scores.keys())
    # Normalise score [-1, 1] to [0, 100] for clean radar rendering
    values = [(s + 1.0) * 50 for s in aspect_scores.values()]
    # Close the polygon loop
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.35)',
        line=dict(color='#3B82F6', width=2.5),
        marker=dict(size=7, color='#60A5FA'),
        name='Aspect Health'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 50, 80],
                ticktext=["Negative", "Neutral", "Positive"],
                linecolor="#334155",
                gridcolor="#334155"
            ),
            angularaxis=dict(
                linecolor="#334155",
                gridcolor="#334155",
                tickfont=dict(size=12, color="#E2E8F0")
            ),
            bgcolor="rgba(15, 23, 42, 0.5)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        height=320,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False
    )
    return fig

def create_net_sentiment_bar(aspect_summary: dict) -> go.Figure:
    """
    Renders diverging horizontal bar chart of Net Sentiment Score per aspect.
    """
    aspects = list(aspect_summary.keys())
    nss_vals = [aspect_summary[a]["net_sentiment_score"] for a in aspects]
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in nss_vals]

    fig = go.Figure(go.Bar(
        x=nss_vals,
        y=aspects,
        orientation='h',
        marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
        text=[f"{v:+.1f}%" for v in nss_vals],
        textposition='outside',
        textfont=dict(color="#F8FAFC", size=12)
    ))

    fig.update_layout(
        title="<b>Net Sentiment Score by Aspect (% Positive - % Negative)</b>",
        title_font=dict(size=15, color="#F8FAFC"),
        xaxis=dict(title="Net Sentiment Score (%)", range=[-105, 105], gridcolor="#334155", zerolinecolor="#64748B"),
        yaxis=dict(gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        height=300,
        margin=dict(l=10, r=30, t=40, b=30)
    )
    return fig

def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders time-series sentiment trend chart over time.
    """
    if "review_date" not in df.columns or "sentiment_score" not in df.columns:
        return None

    df_copy = df.copy()
    df_copy["review_date"] = pd.to_datetime(df_copy["review_date"])
    # Aggregate by date / week
    trend = df_copy.groupby(pd.Grouper(key="review_date", freq="W-MON"))["sentiment_score"].agg(["mean", "count"]).reset_index()
    trend = trend[trend["count"] > 0]

    fig = go.Figure()

    # Moving Average Line
    fig.add_trace(go.Scatter(
        x=trend["review_date"],
        y=trend["mean"],
        mode="lines+markers",
        line=dict(color="#3B82F6", width=3, shape="spline"),
        marker=dict(size=8, color="#60A5FA", line=dict(color="#0F172A", width=1.5)),
        name="Average Sentiment",
        hovertemplate="Week: %{x|%b %d, %Y}<br>Sentiment: %{y:.2f}<extra></extra>"
    ))

    # Reference zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#64748B", annotation_text="Neutral", annotation_position="bottom right")

    fig.update_layout(
        title="<b>Sentiment Trajectory Over Time (Weekly Aggregation)</b>",
        title_font=dict(size=15, color="#F8FAFC"),
        xaxis=dict(title="Date", gridcolor="#334155", tickformat="%b %d"),
        yaxis=dict(title="Mean Polarity Score", range=[-1.05, 1.05], gridcolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=30)
    )
    return fig

def create_emotions_bar(emotions: dict) -> go.Figure:
    """
    Renders bar chart displaying emotional tone distribution.
    """
    labels = list(emotions.keys())
    values = list(emotions.values())
    emo_colors = {
        "Joy": "#10B981",
        "Trust": "#3B82F6",
        "Frustration": "#F59E0B",
        "Disappointment": "#EF4444"
    }
    bar_colors = [emo_colors.get(l, "#8B5CF6") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=bar_colors, line=dict(color="#0F172A", width=1.5)),
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=12)
    ))

    fig.update_layout(
        title="<b>Emotional Tone Profile</b>",
        title_font=dict(size=15, color="#F8FAFC"),
        yaxis=dict(title="Intensity (%)", range=[0, max(values + [50]) * 1.25], gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        height=260,
        margin=dict(l=10, r=10, t=40, b=20)
    )
    return fig

def create_comparison_radar(p1_name: str, p1_scores: dict, p2_name: str, p2_scores: dict) -> go.Figure:
    """
    Side-by-side radar comparison for two competing products.
    """
    categories = list(p1_scores.keys())
    v1 = [(p1_scores[c] + 1.0) * 50 for c in categories]
    v2 = [(p2_scores[c] + 1.0) * 50 for c in categories]

    categories.append(categories[0])
    v1.append(v1[0])
    v2.append(v2[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v1, theta=categories, fill='toself',
        fillcolor='rgba(16, 185, 129, 0.25)',
        line=dict(color='#10B981', width=2),
        name=p1_name
    ))
    fig.add_trace(go.Scatterpolar(
        r=v2, theta=categories, fill='toself',
        fillcolor='rgba(139, 92, 246, 0.25)',
        line=dict(color='#8B5CF6', width=2),
        name=p2_name
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], linecolor="#334155", gridcolor="#334155"),
            angularaxis=dict(linecolor="#334155", gridcolor="#334155", tickfont=dict(size=11, color="#E2E8F0")),
            bgcolor="rgba(15, 23, 42, 0.5)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#CBD5E1")),
        height=350,
        margin=dict(l=30, r=30, t=30, b=40)
    )
    return fig

def generate_wordcloud_image(text_corpus: str, colormap: str = "viridis") -> str:
    """
    Generates a wordcloud and returns base64 PNG data string.
    """
    if not text_corpus or len(text_corpus.strip()) < 10:
        return ""

    wc = WordCloud(
        width=800,
        height=400,
        background_color="#0F172A",
        colormap=colormap,
        max_words=80,
        collocations=False
    ).generate(text_corpus)

    buf = io.BytesIO()
    plt.figure(figsize=(10, 5), facecolor="#0F172A")
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"
