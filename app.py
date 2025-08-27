# app.py
import os
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page config & Theming
# ---------------------------
st.set_page_config(
    page_title="CompeteIQ — AI Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS to mimic a polished dark purple UI
st.markdown(
    """
    <style>
      :root {
        --bg:#0c0714;
        --panel:#17102a;
        --panel-2:#1f163b;
        --accent:#7c4dff;
        --accent-2:#a78bfa;
        --text:#e9e5ff;
        --muted:#b9b1e6;
        --good:#22c55e;
        --bad:#ef4444;
        --warn:#f59e0b;
      }
      .stApp { background: radial-gradient(1200px 800px at 20% -10%, #160f2e 0%, var(--bg) 40%), var(--bg); color: var(--text); }
      section[data-testid="stSidebar"] { background: linear-gradient(180deg, #120a23, #0e0a19); border-right: 1px solid #251b41; }
      .stSidebar [data-testid="stSidebarNav"] { color: var(--text); }
      .st-emotion-cache-16idsys p, .st-emotion-cache-16idsys { color: var(--text) !important; }
      .block-container { padding-top: 1.2rem; }
      .metric-card {
        background: linear-gradient(180deg, var(--panel), var(--panel-2));
        border: 1px solid #271f47;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
      }
      .metric-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }
      .metric-value { color: var(--text); font-size: 1.8rem; font-weight: 700; letter-spacing: 0.2px; }
      .tag {
        display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; margin-left:8px; 
        background:rgba(124,77,255,0.15); border:1px solid rgba(124,77,255,0.35); color:var(--accent-2);
      }
      .pill {
        display:inline-flex; gap:8px; align-items:center; padding:6px 10px; border-radius:999px; 
        background:#1a1330; border:1px solid #2a2150; color:#cfc8ff; font-size: .8rem;
      }
      .section-card {
        background: linear-gradient(180deg, var(--panel), var(--panel-2));
        border: 1px solid #2a2150; border-radius: 18px; padding: 14px 16px;
      }
      h1, h2, h3 { color: var(--text); }
      a, .stMarkdown a { color: var(--accent-2) !important; }
      /* Tables */
      .st-emotion-cache-1y4p8pa { background: transparent; }
      .stDataFrame { border: 1px solid #2a2150; border-radius: 12px; }
      /* Buttons */
      .stButton>button, .stDownloadButton>button {
        background: linear-gradient(180deg, #5533dd, #4b2cc4);
        border: 1px solid #6b4dfd; color: white; border-radius: 12px;
      }
      .stButton>button:hover, .stDownloadButton>button:hover { filter: brightness(1.1); }
      /* Selects/inputs */
      .st-emotion-cache-1jicfl2, .st-emotion-cache-abcdef, .st-emotion-cache-13ln4jf {
        background: var(--panel) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly dark + purple palette
PURPLES = ["#a78bfa", "#8b5cf6", "#7c3aed", "#6d28d9", "#5b21b6", "#4c1d95", "#3b1674"]
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = PURPLES

# ---------------------------
# Utilities
# ---------------------------
LIKELY_DATE_COLS = ["date", "published_at", "timestamp", "datetime", "created_at"]
LIKELY_TEXT_COLS = ["title", "headline", "summary", "content", "text", "description"]
LIKELY_SOURCE_COLS = ["source", "domain", "site", "publisher", "platform"]
LIKELY_TOPIC_COLS = ["company", "brand", "entity", "topic", "tag", "category"]
LIKELY_SENT_COLS = ["sentiment", "polarity", "compound", "score"]

POS_WORDS = set("""
amazing awesome breakthrough beneficial best boost bright celebrate champion clean clear creative delight
effective efficient excellent excite favorable gain good great growth happy improve incredible innovation innovative
leader leading love outperform positive powerful progress robust strong success surge superior win wow
""".split())
NEG_WORDS = set("""
abuse bad bankrupt bug concern crisis critical decline delay dispute drop fail failure fake fear flawed
hack issue layoff loss negative outage poor problem recall risk scandal slowdown slump uncertainty weak worst
""".split())

def find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_cols:
            return lower_cols[c]
    # fuzzy contains
    for c in df.columns:
        for cand in candidates:
            if cand in c.lower():
                return c
    return None

def coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)

def naive_sentiment_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = [t.strip(".,!?()[]{}:;\"'").lower() for t in text.split()]
    pos = sum(t in POS_WORDS for t in tokens)
    neg = sum(t in NEG_WORDS for t in tokens)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(1, (pos + neg))
    return float(np.clip(score, -1.0, 1.0))

def compute_sentiment(df: pd.DataFrame, existing_col: Optional[str], text_cols: List[str]) -> pd.Series:
    if existing_col is not None and pd.api.types.is_numeric_dtype(df[existing_col]):
        # Normalize to [-1, 1] if not already
        s = df[existing_col].astype(float)
        if s.min() >= 0 and s.max() <= 1:
            s = (s - 0.5) * 2
        return s
    # Fallback: compute naive sentiment from best-available text
    for c in text_cols:
        if c in df.columns:
            return df[c].astype(str).fillna("").apply(naive_sentiment_score)
    # If no text, neutral
    return pd.Series([0.0]*len(df), index=df.index)

def locate_csv() -> Optional[Path]:
    preferred = Path("ai_intel_clean.csv")
    alt = Path("/mnt/data/ai_intel_clean.csv")  # provided path in your environment
    if preferred.exists():
        return preferred
    if alt.exists():
        return alt
    # also scan current directory for similarly named files
    for p in Path(".").glob("**/*ai_intel*.csv"):
        return p
    return None

# ---------------------------
# Sidebar — Controls
# ---------------------------
st.sidebar.title("CompeteIQ")
st.sidebar.caption("AI-Powered Market & Competitor Intelligence")
st.sidebar.markdown('<span class="pill">Live <svg height="8" width="8"><circle cx="4" cy="4" r="4" fill="#22c55e"/></svg></span>', unsafe_allow_html=True)

# Data loading (local or upload)
csv_path = locate_csv()
uploaded = None
if csv_path is None:
    st.sidebar.warning("No local CSV found. Upload your dataset to begin.")
    uploaded = st.sidebar.file_uploader("Upload ai_intel_clean.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.DataFrame()
else:
    df = pd.read_csv(csv_path)

# If still empty, stop early but keep the UI
if df.empty:
    st.title("AI Trends Intelligence Dashboard")
    st.info("Upload **ai_intel_clean.csv** from the sidebar to explore trends, sentiment, and sources.")
    st.stop()

# ---------------------------
# Column mapping & enrichment
# ---------------------------
date_col = find_first_col(df, LIKELY_DATE_COLS)
source_col = find_first_col(df, LIKELY_SOURCE_COLS)
topic_col = find_first_col(df, LIKELY_TOPIC_COLS)
sent_col = find_first_col(df, LIKELY_SENT_COLS)

text_candidates = [c for c in LIKELY_TEXT_COLS if c in df.columns]
if not date_col:
    # Create a synthetic date if none exists
    df["__synthetic_date__"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df), freq="D")
    date_col = "__synthetic_date__"

df[date_col] = coerce_date(df[date_col])
df = df.dropna(subset=[date_col])

# Sentiment
df["__sentiment__"] = compute_sentiment(df, sent_col, text_candidates)

# Topic/Brand/entity column fallback
if not topic_col:
    # Try to derive from tags or keywords if exist; else single bucket
    alt_topic_cols = [c for c in df.columns if any(k in c.lower() for k in ["keyword", "tag", "topic", "entity", "brand", "company"])]
    if alt_topic_cols:
        topic_col = alt_topic_cols[0]
    else:
        df["__topic__"] = "AI"
        topic_col = "__topic__"

if not source_col:
    df["__source__"] = "Unknown"
    source_col = "__source__"

# ---------------------------
# Sidebar Filters
# ---------------------------
min_date, max_date = df[date_col].min(), df[date_col].max()
with st.sidebar:
    st.subheader("Filters")
    date_range = st.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(), max_value=max_date.date()
    )
    # Topic filter
    top_topics = df[topic_col].astype(str).value_counts().index.to_list()[:20]
    selected_topics = st.multiselect("Topics / Brands", options=sorted(df[topic_col].astype(str).unique()), default=top_topics[:5])
    # Source filter
    selected_sources = st.multiselect("Sources", options=sorted(df[source_col].astype(str).unique()), default=[])

    st.markdown("---")
    st.caption("Quick Theme")
    _ = st.selectbox("Theme", ["CompeteIQ Dark"], index=0, help="Preset uses purple accent & plotly_dark template")

# Apply filters
mask = (
    (df[date_col].dt.date >= pd.to_datetime(date_range[0]).date()) &
    (df[date_col].dt.date <= pd.to_datetime(date_range[1]).date())
)
if selected_topics:
    mask &= df[topic_col].astype(str).isin(selected_topics)
if selected_sources:
    mask &= df[source_col].astype(str).isin(selected_sources)

fdf = df.loc[mask].copy()

# ---------------------------
# Header / KPIs
# ---------------------------
st.markdown(
    "<h1>Market Trends Analysis <span class='tag'>AI</span></h1>"
    "<p style='color:#b9b1e6;margin-top:-6px;'>Track competitor performance and AI market dynamics.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    with st.container(border=False):
        st.markdown("<div class='metric-card'><div class='metric-label'>Total Mentions</div>"
                    f"<div class='metric-value'>{len(fdf):,}</div></div>", unsafe_allow_html=True)
with col2:
    avg_sent = float(fdf["__sentiment__"].mean()) if not fdf.empty else 0.0
    st.markdown("<div class='metric-card'><div class='metric-label'>Avg Sentiment</div>"
                f"<div class='metric-value'>{avg_sent:.2f}</div></div>", unsafe_allow_html=True)
with col3:
    neg_count = int((fdf["__sentiment__"] < -0.05).sum())
    st.markdown("<div class='metric-card'><div class='metric-label'>Negative Mentions</div>"
                f"<div class='metric-value'>{neg_count}</div></div>", unsafe_allow_html=True)
with col4:
    active_trends = fdf[topic_col].nunique()
    st.markdown("<div class='metric-card'><div class='metric-label'>Active Trends</div>"
                f"<div class='metric-value'>{active_trends}</div></div>", unsafe_allow_html=True)

st.markdown("")

# ---------------------------
# Tabs Layout
# ---------------------------
tab_trends, tab_insights, tab_alerts, tab_sources, tab_data = st.tabs(
    ["📈 Market Trends", "💡 AI Insights", "🔔 Alerts", "🌐 Source Analysis", "🧾 Data"]
)

# --- Trends Tab ---
with tab_trends:
    st.markdown("### Mentions Over Time")
    # Mentions per day
    by_day = fdf.groupby(fdf[date_col].dt.date).size().reset_index(name="Mentions").sort_values(date_col)
    fig_mentions = px.line(
        by_day, x=date_col, y="Mentions",
        markers=True,
        height=360
    )
    fig_mentions.update_traces(hovertemplate="Date=%{x}<br>Mentions=%{y}")
    fig_mentions.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title=None, xaxis_title=None,
        plot_bgcolor="#17102a", paper_bgcolor="#17102a",
    )
    st.plotly_chart(fig_mentions, use_container_width=True)

    st.markdown("### Topic/Brand Trajectory")
    # Top N topics in filtered set
    topN = fdf[topic_col].astype(str).value_counts().head(5).index.tolist()
    ts = (
        fdf[fdf[topic_col].astype(str).isin(topN)]
        .groupby([fdf[date_col].dt.date, topic_col])
        .size()
        .reset_index(name="Mentions")
        .sort_values([date_col, topic_col])
    )
    if not ts.empty:
        fig_topic = px.line(
            ts, x=date_col, y="Mentions", color=topic_col,
            markers=True, height=380
        )
        fig_topic.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#17102a", paper_bgcolor="#17102a",
        )
        st.plotly_chart(fig_topic, use_container_width=True)
    else:
        st.info("Not enough topic data to plot.")

    st.markdown("### Sentiment Distribution")
    hist = px.histogram(
        fdf, x="__sentiment__", nbins=40, height=320,
        labels={"__sentiment__": "Sentiment (−1 to +1)"},
    )
    hist.update_layout(margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#17102a", paper_bgcolor="#17102a")
    st.plotly_chart(hist, use_container_width=True)

# --- Insights Tab ---
with tab_insights:
    st.markdown("### AI-Powered Insights")
    st.caption("Auto-generated summaries from trends & sentiment.")
    insight_card = """
    <div class="section-card" style="margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div><span class="pill">HIGH IMPACT</span></div>
        <div style="color:#9f8cf3;font-size:.8rem;">{ts}</div>
      </div>
      <div style="font-size:1.05rem;margin-top:8px;">
        <b>{title}</b>
      </div>
      <div style="color:#cfc8ff8f;margin-top:6px;">{body}</div>
      <div style="margin-top:10px;">
        {tags}
      </div>
    </div>
    """
    # Simple, data-driven "insights"
    if not fdf.empty:
        top_topics_now = fdf[topic_col].astype(str).value_counts().head(3).index.tolist()
        growth_msg = "Mentions show steady growth over the selected period." if by_day["Mentions"].diff().mean() > 0 else "Mentions remain steady with minor fluctuations."
        pos_trend = "Net sentiment is positive" if avg_sent >= 0.05 else ("Mixed" if -0.05 < avg_sent < 0.05 else "Negative tilt")
        title = f"{top_topics_now[0]} momentum creating a market gap" if top_topics_now else "AI trend momentum update"
        body = f"Analysis of {len(fdf):,} items indicates {growth_msg} {pos_trend.lower()} across coverage. Watchlist topics: {', '.join(top_topics_now)}."
        tags_html = "".join([f"<span class='tag'>{t}</span>" for t in top_topics_now])
        st.markdown(insight_card.format(ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), title=title, body=body, tags=tags_html), unsafe_allow_html=True)

        # Secondary cards from sentiment buckets
        neg_topic = (
            fdf.loc[fdf["__sentiment__"] < -0.05, topic_col]
            .astype(str)
            .value_counts()
            .head(1)
            .index.tolist()
        )
        if neg_topic:
            st.markdown(insight_card.format(
                ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                title=f"Negative sentiment spike detected for {neg_topic[0]}",
                body="Increase in critical coverage may indicate user dissatisfaction or pricing concerns. Consider proactive messaging.",
                tags=f"<span class='tag'>Sentiment</span><span class='tag'>{neg_topic[0]}</span>"
            ), unsafe_allow_html=True)
    else:
        st.info("No insights available for the current filters.")

# --- Alerts Tab ---
with tab_alerts:
    st.markdown("### Alert Management")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='section-card'><b>Launch Keywords</b><br><span class='pill'>Active</span><br><small>Keywords: launch, announcement, release</small></div>", unsafe_allow_html=True)
    with c2:
        thr = float(np.percentile(np.abs(fdf["__sentiment__"]), 85)) if not fdf.empty else 0.3
        st.markdown(f"<div class='section-card'><b>Negative Sentiment Spike</b><br><span class='pill'>Threshold</span> {thr:.2f}</div>", unsafe_allow_html=True)
    with c3:
        vol = int(np.percentile(by_day["Mentions"], 85)) if not fdf.empty else 100
        st.markdown(f"<div class='section-card'><b>High Volume Alert</b><br><span class='pill'>Volume</span> {vol}</div>", unsafe_allow_html=True)

    st.markdown("#### Recent Alerts")
    # Synthetic recent alerts derived from data
    if not fdf.empty and not by_day.empty:
        latest_date = by_day[date_col].max()
        recent_topics = fdf.loc[fdf[date_col].dt.date == latest_date, topic_col].astype(str).value_counts().head(3)
        for t, v in recent_topics.items():
            score = min(10.0, 5 + np.log1p(v) * 2 + max(0, avg_sent) * 3)
            st.markdown(
                f"<div class='section-card' style='margin-bottom:8px;'>"
                f"<span class='pill'>Score {score:.1f}</span> &nbsp; <b>{t}</b>"
                f"<div style='color:#cfc8ff8f;margin-top:4px;'>Mentions surged to {v} on {latest_date}.</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No alerts generated for the current period.")

# --- Source Analysis Tab ---
with tab_sources:
    st.markdown("### Source Leaderboard")
    src = (
        fdf.groupby(source_col)
        .agg(
            Mentions=(source_col, "count"),
            AvgSent=("__sentiment__", "mean"),
        )
        .reset_index()
        .sort_values("Mentions", ascending=False)
        .head(15)
    )
    if not src.empty:
        fig_bar = px.bar(
            src, x="Mentions", y=source_col, orientation="h",
            hover_data={"AvgSent":":.2f"}, height=480
        )
        fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#17102a", paper_bgcolor="#17102a")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Sentiment by Source (Box Plot)")
        fig_box = px.box(
            fdf[[source_col, "__sentiment__"]].dropna(),
            x=source_col, y="__sentiment__", points=False, height=420
        )
        fig_box.update_layout(margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#17102a", paper_bgcolor="#17102a")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No source data available.")

# --- Data Tab ---
with tab_data:
    st.markdown("### Dataset")
    st.caption("Search, scroll and download the filtered view.")
    st.dataframe(
        fdf.sort_values(date_col, ascending=False),
        use_container_width=True,
        height=460,
    )
    csv_bytes = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_bytes,
        file_name="ai_intel_filtered.csv",
        mime="text/csv"
    )

# ---------------------------
# Footer / Status
# ---------------------------
with st.container():
    st.markdown(
        "<div style='opacity:.8;margin-top:12px;'>"
        "<small>Made with ❤️ using Streamlit & Plotly • Theme: CompeteIQ Dark • "
        "If your CSV uses different column names, the app auto-detects the best matches for date, topic, source and sentiment.</small>"
        "</div>",
        unsafe_allow_html=True
    )
