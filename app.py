# =====================================================================
# Credit Card Fraud Detection Dashboard
# Streamlit + Pandas + Scikit-Learn + Plotly (Isolation Forest)
# UI/UX layer only: model loading / prediction logic is unchanged.
# =====================================================================
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FraudSense | Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# 2. DESIGN TOKENS
# ---------------------------------------------------------------------
PRIMARY, SUCCESS, DANGER, WARNING = "#2563EB", "#10B981", "#EF4444", "#F59E0B"
TEXT, MUTED, BORDER = "#111827", "#6B7280", "#E5E7EB"
VERSION = "v2.0.0"

# ---------------------------------------------------------------------
# 3. GLOBAL CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input { font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif; }
.stApp { background: #F8FAFC; color: #111827; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1320px; padding: 1.4rem 2rem 4rem 2rem; }

/* ---------- Top navigation ---------- */
.nav { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;
       background:#fff; border:1px solid #E5E7EB; border-radius:18px; padding:14px 22px;
       box-shadow:0 10px 25px rgba(0,0,0,0.04); margin-bottom:22px; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:40px; height:40px; border-radius:12px; background:#2563EB; color:#fff;
        display:flex; align-items:center; justify-content:center; }
.brand-name { font-weight:700; font-size:18px; letter-spacing:-0.01em; line-height:1.1; }
.brand-sub { font-size:12.5px; color:#6B7280; }
.badges { display:flex; gap:10px; flex-wrap:wrap; }
.badge { display:inline-flex; align-items:center; gap:7px; padding:6px 12px; border-radius:999px;
         font-size:12.5px; font-weight:600; background:#F1F5F9; color:#374151; border:1px solid #E5E7EB; white-space:nowrap; }
.badge.ok { background:#ECFDF5; color:#047857; border-color:#A7F3D0; }
.badge.blue { background:#EFF6FF; color:#1D4ED8; border-color:#BFDBFE; }
.dot { width:7px; height:7px; border-radius:50%; background:#10B981; display:inline-block; }

/* ---------- Hero ---------- */
.hero { background:linear-gradient(120deg,#1E3A8A 0%,#2563EB 100%); color:#fff; border-radius:20px;
        padding:34px 36px; box-shadow:0 10px 25px rgba(37,99,235,0.25); display:flex; flex-wrap:wrap;
        gap:28px; align-items:center; justify-content:space-between; margin-bottom:22px; }
.hero-text { flex:1 1 380px; }
.hero h1 { font-size:clamp(26px,3.2vw,40px); font-weight:800; letter-spacing:-0.03em; line-height:1.12; margin:0 0 10px 0; color:#fff; }
.hero p { color:#DBEAFE; font-size:15.5px; line-height:1.6; margin:0; max-width:520px; }
.hero-stats { flex:1 1 420px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.hero-stat { background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.18); border-radius:14px; padding:14px 16px; }
.hero-stat span { display:block; font-size:12.5px; color:#BFDBFE; margin-bottom:4px; }
.hero-stat b { font-size:21px; font-weight:700; letter-spacing:-0.02em; white-space:nowrap; }

/* ---------- Section headers ---------- */
.sec-title { font-size:20px; font-weight:700; letter-spacing:-0.01em; margin:34px 0 4px 0; color:#111827; }
.sec-sub { font-size:14px; color:#6B7280; margin-bottom:16px; }

/* ---------- KPI cards ---------- */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; }
.kpi { background:#fff; border:1px solid #E5E7EB; border-radius:18px; padding:20px; box-shadow:0 10px 25px rgba(0,0,0,0.05);
       transition:transform .2s ease, box-shadow .2s ease; }
.kpi:hover { transform:translateY(-3px); box-shadow:0 16px 32px rgba(0,0,0,0.08); }
.kpi-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
.kpi-icon { width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center; }
.kpi-label { font-size:13.5px; color:#6B7280; font-weight:500; margin-bottom:4px; }
.kpi-value { font-size:clamp(24px,2.2vw,30px); font-weight:700; letter-spacing:-0.02em; font-variant-numeric:tabular-nums; white-space:nowrap; color:#111827; }
.trend { font-size:12px; font-weight:600; padding:3px 9px; border-radius:999px; white-space:nowrap; }
.t-up-bad { background:#FEF2F2; color:#B91C1C; } .t-good { background:#ECFDF5; color:#047857; }
.t-blue { background:#EFF6FF; color:#1D4ED8; } .t-warn { background:#FFFBEB; color:#B45309; }

/* ---------- Generic card ---------- */
.card { background:#fff; border:1px solid #E5E7EB; border-radius:18px; padding:22px; box-shadow:0 10px 25px rgba(0,0,0,0.05); }

/* ---------- Upload ---------- */
.upload-head { display:flex; align-items:center; gap:16px; background:#fff; border:1px solid #E5E7EB; border-bottom:none;
               border-radius:18px 18px 0 0; padding:22px 24px 8px 24px; }
.upload-icon { width:48px; height:48px; border-radius:14px; background:#EFF6FF; color:#2563EB; display:flex; align-items:center; justify-content:center; }
.upload-head b { display:block; font-size:16px; } .upload-head span { font-size:13.5px; color:#6B7280; }
[data-testid="stFileUploader"] { background:#fff; border:1px solid #E5E7EB; border-top:none; border-radius:0 0 18px 18px;
                                 padding:6px 24px 24px 24px; box-shadow:0 10px 25px rgba(0,0,0,0.05); }
[data-testid="stFileUploader"] label { display:none; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed #BFDBFE; background:#F8FAFF; border-radius:14px; padding:26px; transition:all .2s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#2563EB; background:#EFF6FF; }
[data-testid="stFileUploaderDropzone"] button { border-radius:10px; border:1px solid #2563EB; color:#2563EB; background:#fff; font-weight:600; }

/* ---------- Charts ---------- */
div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #E5E7EB; border-radius:18px; padding:10px 14px;
                                   box-shadow:0 10px 25px rgba(0,0,0,0.05); transition:box-shadow .2s ease; }
div[data-testid="stPlotlyChart"]:hover { box-shadow:0 16px 32px rgba(0,0,0,0.08); }

/* ---------- Tables ---------- */
.tbl-wrap { background:#fff; border:1px solid #E5E7EB; border-radius:18px; box-shadow:0 10px 25px rgba(0,0,0,0.05);
            max-height:430px; overflow:auto; }
.tbl { width:100%; border-collapse:separate; border-spacing:0; font-size:13.5px; font-variant-numeric:tabular-nums; }
.tbl thead th { position:sticky; top:0; z-index:2; background:#F9FAFB; color:#6B7280; font-weight:600; text-align:left;
                padding:12px 16px; border-bottom:1px solid #E5E7EB; white-space:nowrap; }
.tbl td { padding:11px 16px; border-bottom:1px solid #F3F4F6; white-space:nowrap; color:#111827; }
.tbl tbody tr:nth-child(even) { background:#FAFBFC; }
.tbl tbody tr { transition:background .15s ease; } .tbl tbody tr:hover { background:#EFF6FF; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
.p-red { background:#FEF2F2; color:#B91C1C; } .p-green { background:#ECFDF5; color:#047857; }
.p-amber { background:#FFFBEB; color:#B45309; } .p-dark { background:#7F1D1D; color:#fff; }
.riskbar { display:inline-block; width:90px; height:7px; background:#F3F4F6; border-radius:99px; vertical-align:middle; margin-right:10px; overflow:hidden; }
.riskbar i { display:block; height:100%; border-radius:99px; }

/* ---------- Summary + alerts ---------- */
.sum-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:16px; }
.sum { background:#fff; border:1px solid #E5E7EB; border-radius:18px; padding:18px 20px; box-shadow:0 10px 25px rgba(0,0,0,0.05); transition:transform .2s ease; }
.sum:hover { transform:translateY(-3px); }
.sum span { font-size:13px; color:#6B7280; display:block; margin-bottom:6px; }
.sum b { font-size:22px; letter-spacing:-0.02em; white-space:nowrap; }
.alert-note { display:flex; gap:12px; align-items:center; background:#FEF2F2; border:1px solid #FECACA; color:#991B1B;
              border-radius:14px; padding:12px 16px; font-size:14px; margin-bottom:14px; }

/* ---------- Export ---------- */
.export { display:flex; align-items:center; gap:16px; background:#fff; border:1px solid #E5E7EB; border-radius:18px 18px 0 0;
          border-bottom:none; padding:24px; box-shadow:0 10px 25px rgba(0,0,0,0.05); }
.export b { font-size:17px; display:block; margin-bottom:3px; } .export span { color:#6B7280; font-size:14px; }
.stDownloadButton { background:#fff; border:1px solid #E5E7EB; border-top:none; border-radius:0 0 18px 18px; padding:0 24px 24px 24px; margin-top:-1px; }
.stDownloadButton button { width:100%; background:#2563EB; color:#fff; border:none; border-radius:12px; padding:0.85rem 1.4rem;
                           font-size:16px; font-weight:600; box-shadow:0 8px 18px rgba(37,99,235,0.28); transition:all .2s ease; }
.stDownloadButton button:hover { background:#1D4ED8; color:#fff; transform:translateY(-2px); box-shadow:0 12px 24px rgba(37,99,235,0.35); }
.stDownloadButton button:focus-visible { outline:3px solid #BFDBFE; outline-offset:2px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #E5E7EB; }
[data-testid="stSidebar"] .block-container { padding:1.6rem 1.2rem; }
.sb-sec { margin-bottom:22px; }
.sb-title { font-size:12.5px; font-weight:700; color:#6B7280; margin-bottom:10px; letter-spacing:0.02em; }
.sb-row { display:flex; justify-content:space-between; font-size:13.5px; padding:7px 0; border-bottom:1px solid #F3F4F6; gap:10px; }
.sb-row span { color:#6B7280; } .sb-row b { font-weight:600; text-align:right; }
.chips { display:flex; flex-wrap:wrap; gap:6px; }
.chip { background:#F1F5F9; color:#374151; border-radius:8px; padding:4px 10px; font-size:12.5px; font-weight:500; }

.empty { text-align:center; padding:44px 24px; margin-top:26px; background:#fff; border:1px solid #E5E7EB; border-radius:18px; box-shadow:0 10px 25px rgba(0,0,0,0.05); }
.empty b { font-size:18px; display:block; margin-bottom:6px; } .empty span { color:#6B7280; font-size:14.5px; }
.foot { text-align:center; color:#9CA3AF; font-size:13px; margin-top:44px; }

@media (max-width: 900px) {
  .block-container { padding:1rem 1rem 3rem 1rem; }
  .hero { padding:24px 22px; } .hero-stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# 4. HELPERS
# ---------------------------------------------------------------------
ICONS = {
    "layers": "M12 2 2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5",
    "shield": "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z M12 8v4 M12 16h.01",
    "check": "M22 11.08V12a10 10 0 1 1-5.93-9.14 M22 4 12 14.01l-3-3",
    "percent": "M19 5 5 19 M6.5 9a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z M17.5 20a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z",
    "clock": "M12 6v6l4 2 M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
    "cpu": "M4 4h16v16H4z M9 9h6v6H9z M9 2v2 M15 2v2 M9 20v2 M15 20v2 M2 9h2 M2 15h2 M20 9h2 M20 15h2",
    "upload": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
    "download": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
    "alert": "M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01",
}


def icon(name, size=20):
    """Inline SVG icon (stroke follows the parent's text color)."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="{ICONS[name]}"/></svg>')


def html(markup):
    """Flatten markup so Streamlit's markdown parser never treats it as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


def section(title, sub=""):
    st.markdown(html(f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div>'), unsafe_allow_html=True)


def style_fig(fig, title, height=380):
    """Shared professional Plotly theme."""
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=16, color=TEXT), x=0.02, y=0.96),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=13, color=MUTED), height=height,
        margin=dict(t=64, b=28, l=24, r=24),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", font=dict(color=TEXT)),
        transition=dict(duration=500, easing="cubic-in-out"),
        hoverlabel=dict(bgcolor="white", font=dict(family="Inter", color=TEXT), bordercolor=BORDER),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#F3F4F6", zeroline=False, tickfont=dict(color=MUTED))
    return fig


def to_table(frame, flag_col=None):
    """Render a DataFrame as a styled HTML table (sticky header, hover, zebra rows)."""
    frame = frame.copy()
    if flag_col and flag_col in frame.columns:
        frame[flag_col] = frame[flag_col].map({
            "Fraud": '<span class="pill p-red">Fraud</span>',
            "Normal": '<span class="pill p-green">Normal</span>',
        })
    body = frame.to_html(classes="tbl", border=0, escape=False,
                         float_format=lambda v: f"{v:,.4f}")
    return f'<div class="tbl-wrap">{body}</div>'


def risk_level(rate):
    if rate < 2:
        return "Low", SUCCESS
    if rate < 5:
        return "Moderate", WARNING
    return "High", DANGER


# ---------------------------------------------------------------------
# 5. TOP NAVIGATION
# ---------------------------------------------------------------------
st.markdown(html(f"""
<div class="nav">
  <div class="brand">
    <div class="logo">{icon("shield", 22)}</div>
    <div><div class="brand-name">FraudSense</div><div class="brand-sub">Fraud Detection Console</div></div>
  </div>
  <div class="badges">
    <span class="badge ok"><span class="dot"></span>All systems operational</span>
    <span class="badge blue">Model: Isolation Forest</span>
    <span class="badge">Updated {datetime.now().strftime("%d %b %Y, %H:%M")}</span>
  </div>
</div>
"""), unsafe_allow_html=True)

# Placeholder for the hero (filled once we know the results)
hero_slot = st.empty()

# ---------------------------------------------------------------------
# 6. UPLOAD
# ---------------------------------------------------------------------
st.markdown(html(f"""
<div class="upload-head">
  <div class="upload-icon">{icon("upload", 24)}</div>
  <div><b>Upload Transaction CSV</b><span>Drag and drop a file below, or browse. Accepted: .csv</span></div>
</div>
"""), unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"], label_visibility="collapsed")

stats = None  # quick stats for the sidebar

if uploaded_file is not None:

    # ---------------- ML logic (unchanged) ----------------
    df = pd.read_csv(uploaded_file)

    feature_names = joblib.load("models/feature_names.pkl")
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load("models/isolation_forest.pkl")

    t_start = time.perf_counter()

    X = df[feature_names]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    df["Prediction"] = ["Fraud" if p == -1 else "Normal" for p in predictions]

    fraud_count = (df["Prediction"] == "Fraud").sum()
    normal_count = (df["Prediction"] == "Normal").sum()
    total_count = len(df)
    fraud_percentage = round(fraud_count / total_count * 100, 2)
    # --------------------------------------------------------

    # Presentation-only metrics derived from the model output
    scores = model.decision_function(X_scaled)               # < 0 means anomaly
    risk = np.clip(50 - scores * 200, 0, 100)                # 0-100 risk score
    proc_time = time.perf_counter() - t_start
    level, level_color = risk_level(fraud_percentage)

    if "Class" in df.columns:                                # accuracy only if labels exist
        accuracy = f'{((df["Prediction"] == "Fraud") == (df["Class"] == 1)).mean() * 100:.2f}%'
    else:
        accuracy = "N/A"

    completeness = 100 - df.isna().sum().sum() / df.size * 100
    low_n = int((scores >= 0.05).sum())
    med_n = int(((scores >= 0) & (scores < 0.05)).sum())
    high_n = int((scores < 0).sum())
    stats = dict(total=total_count, fraud=fraud_count, rate=fraud_percentage, cols=len(feature_names))

    # ---------------- Hero ----------------
    hero_slot.markdown(html(f"""
    <div class="hero">
      <div class="hero-text">
        <h1>Credit Card Fraud Detection System</h1>
        <p>Unsupervised anomaly detection that screens every transaction and surfaces the ones that need review.</p>
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><span>Model</span><b>Isolation Forest</b></div>
        <div class="hero-stat"><span>Detection accuracy</span><b>{accuracy}</b></div>
        <div class="hero-stat"><span>Records processed</span><b>{total_count:,}</b></div>
        <div class="hero-stat"><span>Fraud risk level</span><b>{level}</b></div>
      </div>
    </div>"""), unsafe_allow_html=True)

    # ---------------- KPI cards ----------------
    section("Dashboard Overview", "Key metrics for the file you uploaded.")

    def kpi(ic, bg, fg, label, value, chip, chip_cls):
        return (f'<div class="kpi"><div class="kpi-top"><div class="kpi-icon" style="background:{bg};color:{fg};">'
                f'{icon(ic, 20)}</div><span class="trend {chip_cls}">{chip}</span></div>'
                f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>')

    st.markdown(html('<div class="kpi-grid">' + "".join([
        kpi("layers", "#EFF6FF", PRIMARY, "Total Transactions", f"{total_count:,}", "100% scanned", "t-blue"),
        kpi("shield", "#FEF2F2", DANGER, "Fraud Transactions", f"{fraud_count:,}", f"▲ {fraud_percentage}% of volume", "t-up-bad"),
        kpi("check", "#ECFDF5", SUCCESS, "Normal Transactions", f"{normal_count:,}", f"▼ {100 - fraud_percentage:.2f}% of volume", "t-good"),
        kpi("percent", "#FFFBEB", WARNING, "Fraud Rate", f"{fraud_percentage}%", f"{level} risk", "t-warn"),
        kpi("clock", "#EFF6FF", PRIMARY, "Processing Time", f"{proc_time:.2f}s", f"{total_count / max(proc_time, 1e-6):,.0f} rows/s", "t-blue"),
        kpi("cpu", "#ECFDF5", SUCCESS, "Model Status", "Active", "● Healthy", "t-good"),
    ]) + "</div>"), unsafe_allow_html=True)

    # ---------------- Charts ----------------
    section("Fraud Analytics", "How flagged transactions are distributed.")

    chart_data = pd.DataFrame({"Category": ["Fraud", "Normal"], "Count": [fraud_count, normal_count]})
    cmap = {"Fraud": DANGER, "Normal": SUCCESS}

    # Chart 1: donut
    donut = px.pie(chart_data, names="Category", values="Count", hole=0.72, color="Category", color_discrete_map=cmap)
    donut.update_traces(sort=False, textinfo="none", marker=dict(line=dict(color="white", width=4)),
                        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>")
    donut.add_annotation(text=f"<b>{fraud_percentage}%</b><br><span style='font-size:12px;color:{MUTED}'>Fraud</span>",
                         showarrow=False, font=dict(size=32, color=TEXT, family="Inter"))
    style_fig(donut, "Fraud vs Normal")

    # Chart 2: bar
    bar = px.bar(chart_data, x="Category", y="Count", color="Category", text="Count", color_discrete_map=cmap)
    bar.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False, marker_line_width=0,
                      hovertemplate="%{x}: %{y:,}<extra></extra>")
    try:
        bar.update_traces(marker_cornerradius=10)            # rounded bars (Plotly >= 5.19)
    except Exception:
        pass
    style_fig(bar, "Fraud Distribution")
    bar.update_layout(showlegend=False, bargap=0.5)
    bar.update_xaxes(title=None)
    bar.update_yaxes(title=None)

    # Chart 3: trend (by Time if present, otherwise by record order)
    if "Time" in df.columns:
        seg = (df["Time"] // 3600).astype(int)
        x_title = "Hours since first transaction"
    else:
        seg = pd.cut(np.arange(total_count), bins=24, labels=False)
        x_title = "Record segment (file order)"
    trend = (df.assign(_seg=seg).groupby("_seg")["Prediction"].apply(lambda s: (s == "Fraud").sum()).reset_index())
    trend.columns = ["Segment", "Fraud"]
    area = go.Figure(go.Scatter(x=trend["Segment"], y=trend["Fraud"], mode="lines", fill="tozeroy",
                                line=dict(color=PRIMARY, width=3, shape="spline"), fillcolor="rgba(37,99,235,0.10)",
                                hovertemplate="%{x}: %{y:,} flagged<extra></extra>"))
    style_fig(area, "Fraud Trend Analysis")
    area.update_layout(showlegend=False)
    area.update_xaxes(title=x_title)
    area.update_yaxes(title=None)

    # Chart 4: risk breakdown (horizontal)
    rb = pd.DataFrame({"Level": ["Low Risk", "Medium Risk", "High Risk"], "Count": [low_n, med_n, high_n]})
    rbar = px.bar(rb, y="Level", x="Count", orientation="h", color="Level", text="Count",
                  color_discrete_map={"Low Risk": SUCCESS, "Medium Risk": WARNING, "High Risk": DANGER})
    rbar.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False, marker_line_width=0,
                       hovertemplate="%{y}: %{x:,}<extra></extra>")
    try:
        rbar.update_traces(marker_cornerradius=8)
    except Exception:
        pass
    style_fig(rbar, "Risk Breakdown")
    rbar.update_layout(showlegend=False, bargap=0.45)
    rbar.update_yaxes(title=None, autorange="reversed", showgrid=False)
    rbar.update_xaxes(title=None, showgrid=True, gridcolor="#F3F4F6")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(donut, use_container_width=True)
    with c2:
        st.plotly_chart(bar, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(area, use_container_width=True)
    with c4:
        st.plotly_chart(rbar, use_container_width=True)

    # ---------------- Executive summary ----------------
    section("Executive Summary", "Overall health of the model and the dataset.")

    def summ(label, value, color=TEXT):
        return f'<div class="sum"><span>{label}</span><b style="color:{color};">{value}</b></div>'

    st.markdown(html('<div class="sum-grid">' + "".join([
        summ("Total risk score", f"{risk.mean():.1f} / 100", level_color),
        summ("Fraud percentage", f"{fraud_percentage}%", DANGER),
        summ("Model health", "Operational", SUCCESS),
        summ("Dataset health", f"{completeness:.1f}% complete", SUCCESS if completeness > 99 else WARNING),
        summ("Processing status", f"Completed in {proc_time:.2f}s", PRIMARY),
    ]) + "</div>"), unsafe_allow_html=True)

    # ---------------- Fraud alerts ----------------
    section("🚨 Fraud Alerts", "The ten highest-risk transactions the model flagged.")

    fraud_df = df[df["Prediction"] == "Fraud"]

    if len(fraud_df) > 0:
        st.markdown(html(f'<div class="alert-note">{icon("alert", 20)}<span><b>{fraud_count:,} transactions</b> '
                         f'need review. Showing the top 10 by risk score.</span></div>'), unsafe_allow_html=True)

        top = pd.DataFrame({"risk": risk}, index=df.index).loc[fraud_df.index].nlargest(10, "risk")
        rows = ""
        for idx, r in top["risk"].items():
            if r >= 80:
                lvl, cls, col = "Critical", "p-dark", "#7F1D1D"
            elif r >= 65:
                lvl, cls, col = "High", "p-red", DANGER
            else:
                lvl, cls, col = "Elevated", "p-amber", WARNING
            amount = f'{df.loc[idx, "Amount"]:,.2f}' if "Amount" in df.columns else "N/A"
            rows += (f'<tr><td>#{idx}</td><td>{amount}</td>'
                     f'<td><span class="riskbar"><i style="width:{r:.0f}%;background:{col};"></i></span>{r:.1f}</td>'
                     f'<td><span class="pill {cls}">{lvl}</span></td></tr>')
        st.markdown(html('<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Transaction</th><th>Amount</th>'
                         f'<th>Risk score</th><th>Alert level</th></tr></thead><tbody>{rows}</tbody></table></div>'),
                    unsafe_allow_html=True)
    else:
        st.success("No Fraud Transactions Detected")

    # ---------------- Tables ----------------
    section("📄 Uploaded Dataset", "First 50 rows with the model's prediction.")
    st.markdown(to_table(df.head(50), "Prediction"), unsafe_allow_html=True)

    section("All Flagged Transactions", "First 100 transactions classified as fraud.")
    if len(fraud_df) > 0:
        st.markdown(to_table(fraud_df.head(100), "Prediction"), unsafe_allow_html=True)
    else:
        st.success("No Fraud Transactions Detected")

    # ---------------- Export ----------------
    section("Export", "Take the results with you.")
    csv = df.to_csv(index=False).encode("utf-8")
    st.markdown(html(f"""
    <div class="export">
      <div class="upload-icon">{icon("download", 24)}</div>
      <div><b>Prediction results</b><span>{total_count:,} rows · {fraud_count:,} flagged as fraud · CSV format</span></div>
    </div>"""), unsafe_allow_html=True)
    st.download_button(label="Download Prediction Results", data=csv,
                       file_name="fraud_detection_results.csv", mime="text/csv", use_container_width=True)

else:
    # ---------------- Empty state (before upload) ----------------
    hero_slot.markdown(html("""
    <div class="hero">
      <div class="hero-text">
        <h1>Credit Card Fraud Detection System</h1>
        <p>Unsupervised anomaly detection that screens every transaction and surfaces the ones that need review.</p>
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><span>Model</span><b>Isolation Forest</b></div>
        <div class="hero-stat"><span>Detection accuracy</span><b>—</b></div>
        <div class="hero-stat"><span>Records processed</span><b>—</b></div>
        <div class="hero-stat"><span>Fraud risk level</span><b>—</b></div>
      </div>
    </div>"""), unsafe_allow_html=True)
    st.markdown(html("""
    <div class="empty"><b>No data yet</b>
    <span>Upload a transaction CSV above to see KPIs, charts, fraud alerts and exports.</span></div>"""),
                unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 7. FOOTER
# ---------------------------------------------------------------------
st.markdown(f'<div class="foot">FraudSense {VERSION} · Powered by Isolation Forest, Streamlit and Plotly</div>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 8. SIDEBAR
# ---------------------------------------------------------------------
def row(k, v):
    return f'<div class="sb-row"><span>{k}</span><b>{v}</b></div>'


quick = (row("Transactions", f"{stats['total']:,}") + row("Flagged", f"{stats['fraud']:,}") +
         row("Fraud rate", f"{stats['rate']}%") + row("Features used", stats["cols"])
         if stats else '<div class="sb-row"><span>Upload a file to see statistics</span></div>')

with st.sidebar:
    st.markdown(html(f"""
    <div class="brand" style="margin-bottom:24px;"><div class="logo">{icon("shield", 22)}</div>
      <div><div class="brand-name">FraudSense</div><div class="brand-sub">{VERSION}</div></div></div>
    <div class="sb-sec"><div class="sb-title">Project information</div>
      {row("Project", "Fraud Detection")}{row("Type", "Anomaly detection")}{row("Status", "Production demo")}</div>
    <div class="sb-sec"><div class="sb-title">Model information</div>
      {row("Algorithm", "Isolation Forest")}{row("Learning", "Unsupervised")}{row("Library", "Scikit-Learn")}</div>
    <div class="sb-sec"><div class="sb-title">Technology stack</div>
      <div class="chips"><span class="chip">Python</span><span class="chip">Streamlit</span><span class="chip">Pandas</span>
      <span class="chip">Scikit-Learn</span><span class="chip">Plotly</span></div></div>
    <div class="sb-sec"><div class="sb-title">Quick statistics</div>{quick}</div>
    <div class="sb-sec"><div class="sb-title">Developer</div>
      {row("Built by", "Aditya")}{row("Version", VERSION)}</div>
    """), unsafe_allow_html=True)