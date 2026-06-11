import streamlit as st
import pandas as pd
import numpy as np
import pickle
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from datetime import datetime
import heapq
from location_log import log_location, get_log, get_today_summary, get_last_location

# ── PAGE CONFIG ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── DEEP PURPLE ATMOSPHERE ── */
.stApp {
    background:
        radial-gradient(ellipse at 0% 0%,   rgba(88,28,135,0.55)  0%, transparent 50%),
        radial-gradient(ellipse at 100% 0%,  rgba(109,40,217,0.35) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 50%,  rgba(76,29,149,0.25)  0%, transparent 60%),
        radial-gradient(ellipse at 0% 100%,  rgba(55,20,110,0.45)  0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%,rgba(88,28,135,0.3)   0%, transparent 45%),
        radial-gradient(ellipse at 50% 100%, rgba(30,10,60,0.6)    0%, transparent 55%),
        linear-gradient(160deg, #0D0414 0%, #120820 30%, #0A0512 60%, #080310 100%);
    min-height: 100vh;
}

/* ── STAR PARTICLES ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(1px 1px at 8%  15%, rgba(220,180,255,0.5)  0%, transparent 100%),
        radial-gradient(1px 1px at 22% 6%,  rgba(255,255,255,0.3)  0%, transparent 100%),
        radial-gradient(1px 1px at 38% 22%, rgba(200,150,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 52% 9%,  rgba(255,255,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 68% 18%, rgba(220,180,255,0.3)  0%, transparent 100%),
        radial-gradient(1px 1px at 80% 5%,  rgba(255,255,255,0.4)  0%, transparent 100%),
        radial-gradient(1px 1px at 91% 28%, rgba(200,150,255,0.3)  0%, transparent 100%),
        radial-gradient(1px 1px at 15% 48%, rgba(255,255,255,0.2)  0%, transparent 100%),
        radial-gradient(1px 1px at 33% 58%, rgba(220,180,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 47% 72%, rgba(255,255,255,0.18) 0%, transparent 100%),
        radial-gradient(1px 1px at 72% 55%, rgba(200,150,255,0.22) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 75%, rgba(255,255,255,0.2)  0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 4%  82%, rgba(220,180,255,0.35) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 93% 62%, rgba(255,255,255,0.28) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 58% 42%, rgba(200,150,255,0.2)  0%, transparent 100%),
        radial-gradient(2px 2px at 75% 88%,  rgba(180,120,255,0.3)  0%, transparent 100%),
        radial-gradient(2px 2px at 28% 92%,  rgba(255,255,255,0.2)  0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── TITLE ── */
h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px !important;
    color: #FFFFFF !important;
    line-height: 1.1 !important;
}

/* ── SUBTITLE ── */
.stApp [data-testid="stCaptionContainer"] p {
    color: rgba(200,160,255,0.45) !important;
    font-size: 0.72rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg,
        transparent,
        rgba(160,100,255,0.2),
        rgba(180,120,255,0.15),
        transparent
    ) !important;
    margin: 6px 0 20px 0 !important;
}

/* ══════════════════════════════════════
   LIQUID GLASS TABS — DEEP PURPLE
══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(88,28,135,0.12) !important;
    border: 1px solid rgba(180,120,255,0.18) !important;
    border-radius: 22px !important;
    padding: 6px !important;
    gap: 4px !important;
    backdrop-filter: blur(60px) saturate(200%) brightness(1.1) !important;
    -webkit-backdrop-filter: blur(60px) saturate(200%) brightness(1.1) !important;
    box-shadow:
        inset 0 1px 0 rgba(200,150,255,0.2),
        inset 0 -1px 0 rgba(0,0,0,0.35),
        inset 1px 0 0 rgba(180,120,255,0.08),
        inset -1px 0 0 rgba(180,120,255,0.08),
        0 8px 40px rgba(88,28,135,0.35),
        0 2px 12px rgba(0,0,0,0.4) !important;
    position: relative !important;
}

.stTabs [data-baseweb="tab-list"]::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: 0 !important;
    right: 0 !important; bottom: 0 !important;
    border-radius: 22px !important;
    background: linear-gradient(
        135deg,
        rgba(200,150,255,0.1) 0%,
        rgba(120,60,200,0.04) 35%,
        rgba(88,28,135,0.06) 65%,
        rgba(160,100,255,0.08) 100%
    ) !important;
    pointer-events: none !important;
    z-index: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 16px !important;
    color: rgba(200,160,255,0.5) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 9px 20px !important;
    border: none !important;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    letter-spacing: 0.2px !important;
    position: relative !important;
    background: transparent !important;
    z-index: 1 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: rgba(220,190,255,0.85) !important;
    background: rgba(160,100,255,0.08) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow:
        inset 0 1px 0 rgba(200,150,255,0.2),
        0 2px 12px rgba(88,28,135,0.2) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(139,92,246,0.22) !important;
    color: #FFFFFF !important;
    border-bottom: none !important;
    backdrop-filter: blur(40px) saturate(220%) brightness(1.15) !important;
    -webkit-backdrop-filter: blur(40px) saturate(220%) brightness(1.15) !important;
    box-shadow:
        inset 0 1.5px 0 rgba(220,180,255,0.55),
        inset 0 -1px 0 rgba(60,20,100,0.5),
        inset 1px 0 0 rgba(180,130,255,0.25),
        inset -1px 0 0 rgba(180,130,255,0.25),
        0 6px 28px rgba(109,40,217,0.4),
        0 2px 8px rgba(88,28,135,0.5) !important;
    border: 1px solid rgba(180,130,255,0.3) !important;
}

.stTabs [aria-selected="true"]::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 12% !important;
    width: 76% !important;
    height: 1px !important;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(230,200,255,0.75),
        transparent
    ) !important;
    border-radius: 1px !important;
    z-index: 2 !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    border: 1px solid rgba(180,130,255,0.3) !important;
    border-radius: 12px !important;
    padding: 11px 22px !important;
    box-shadow:
        inset 0 1px 0 rgba(220,180,255,0.3),
        0 4px 24px rgba(109,40,217,0.45) !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #8B5CF6, #7C3AED) !important;
    transform: translateY(-2px) !important;
    box-shadow:
        inset 0 1px 0 rgba(220,180,255,0.4),
        0 8px 32px rgba(109,40,217,0.6) !important;
    border-color: rgba(200,160,255,0.4) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    background: linear-gradient(135deg, #6D28D9, #5B21B6) !important;
    box-shadow: 0 2px 12px rgba(109,40,217,0.35) !important;
}

/* ── SUBHEADERS ── */
h2, h3 {
    color: rgba(235,220,255,0.9) !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px !important;
}

/* ── BODY TEXT ── */
.stMarkdown p {
    color: rgba(200,170,255,0.7) !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}

.stMarkdown strong {
    color: rgba(235,220,255,0.92) !important;
    font-weight: 600 !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: rgba(88,28,135,0.1) !important;
    border: 1px solid rgba(160,100,255,0.15) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    backdrop-filter: blur(20px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
    box-shadow:
        inset 0 1px 0 rgba(200,150,255,0.12),
        0 4px 20px rgba(88,28,135,0.2) !important;
    transition: all 0.25s ease !important;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(180,130,255,0.35) !important;
    background: rgba(109,40,217,0.15) !important;
    box-shadow:
        inset 0 1px 0 rgba(200,150,255,0.2),
        0 6px 28px rgba(109,40,217,0.3) !important;
}

[data-testid="stMetricLabel"] p {
    color: rgba(180,140,255,0.5) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: #EDE9FE !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    padding: 14px 18px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: none !important;
}

.stSuccess {
    background: rgba(16,185,129,0.08) !important;
    border-left: 3px solid #10B981 !important;
    border-radius: 0 12px 12px 0 !important;
}

.stWarning {
    background: rgba(245,158,11,0.08) !important;
    border-left: 3px solid #F59E0B !important;
    border-radius: 0 12px 12px 0 !important;
}

.stError {
    background: rgba(239,68,68,0.08) !important;
    border-left: 3px solid #EF4444 !important;
    border-radius: 0 12px 12px 0 !important;
}

.stInfo {
    background: rgba(139,92,246,0.08) !important;
    border-left: 3px solid rgba(139,92,246,0.5) !important;
    border-radius: 0 12px 12px 0 !important;
    color: rgba(200,170,255,0.7) !important;
}

/* ── PROGRESS BARS ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7C3AED, #A78BFA) !important;
    border-radius: 4px !important;
}

.stProgress > div > div {
    background: rgba(139,92,246,0.12) !important;
    border-radius: 4px !important;
    height: 6px !important;
}

/* ── SELECTBOX ── */
.stSelectbox > label {
    color: rgba(180,140,255,0.5) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 500 !important;
}

.stSelectbox [data-baseweb="select"] > div {
    background: rgba(88,28,135,0.12) !important;
    border: 1px solid rgba(160,100,255,0.15) !important;
    border-radius: 12px !important;
    color: rgba(220,200,255,0.85) !important;
    font-size: 0.88rem !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    transition: border-color 0.2s ease !important;
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: rgba(180,130,255,0.4) !important;
    background: rgba(109,40,217,0.15) !important;
}

/* ── TOGGLE ── */
.stToggle label {
    color: rgba(200,170,255,0.6) !important;
    font-size: 0.85rem !important;
}

/* ── CAPTION ── */
.stCaption, caption {
    color: rgba(160,120,255,0.4) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.3px !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: #8B5CF6 !important;
}

/* ── HORIZONTAL SPACING ── */
[data-testid="stHorizontalBlock"] {
    gap: 16px !important;
}

/* ── PLOTLY CHARTS ── */
.js-plotly-plot {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* ── MAIN BLOCK PADDING ── */
.main .block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(139,92,246,0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(139,92,246,0.6);
}
</style>
""", unsafe_allow_html=True)


# ── LOCATION GRAPH (edit distances here for college) ──────────
LOCATION_GRAPH = {
    'my_desk':    {'rahul_desk': 5,  'main_door': 15},
    'rahul_desk': {'my_desk': 5,     'aadhil': 4,    'main_door': 12},
    'aadhil':     {'rahul_desk': 4,  'main_door': 8},
    'main_door':  {'my_desk': 15,    'rahul_desk': 12, 'aadhil': 8}
}

LOCATION_LABELS = {
    'my_desk':    'My Desk',
    'rahul_desk': "Rahul's Desk",
    'aadhil':     "Aadhil's Desk",
    'main_door':  'Main Door'
}
# Human readable turn-by-turn directions for each connection
DIRECTIONS = {
    ('main_door',  'aadhil'):     "Enter and walk straight ahead",
    ('aadhil',     'rahul_desk'): "Take a left and walk to the next desk",
    ('aadhil',     'main_door'):  "Turn around and walk to the exit",
    ('rahul_desk', 'aadhil'):     "Walk straight to the desk ahead",
    ('rahul_desk', 'my_desk'):    "Turn right and walk to the corner desk",
    ('rahul_desk', 'main_door'):  "Walk back toward the entrance",
    ('my_desk',    'rahul_desk'): "Walk from the corner toward the center",
    ('my_desk',    'main_door'):  "Walk straight toward the exit",
    ('main_door',  'rahul_desk'): "Enter and walk to the second desk on left",
}

# ── DIJKSTRA ──────────────────────────────────────────────────
def dijkstra(graph, start, end):
    queue = [(0, start, [start])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            return path, cost
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
    return None, float('inf')

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model/pathfi_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('model/feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, le, features

# ── WIFI SCAN ─────────────────────────────────────────────────
def scan_wifi():
    result = subprocess.run(
        ["netsh", "wlan", "show", "networks", "mode=bssid"],
        capture_output=True, text=True
    )
    networks = {}
    lines = result.stdout.split('\n')
    current_ssid = None
    current_bssid = None
    for line in lines:
        line = line.strip()
        if line.startswith("SSID") and "BSSID" not in line:
            parts = line.split(":")
            if len(parts) > 1:
                current_ssid = parts[1].strip()
        elif line.startswith("BSSID"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                current_bssid = parts[1].strip()
        elif line.startswith("Signal"):
            parts = line.split(":")
            if len(parts) > 1 and current_ssid and current_bssid:
                signal_percent = int(parts[1].strip().replace('%', ''))
                rssi = (signal_percent / 2) - 100
                col_name = f"rssi_{current_ssid}_{current_bssid[-5:]}"
                networks[col_name] = rssi
                current_ssid = None
                current_bssid = None
    return networks

# ── PREDICT LOCATION ──────────────────────────────────────────
def predict_location():
    model, le, feature_columns = load_model()
    networks = scan_wifi()
    row = {col: networks.get(col, -100) for col in feature_columns}
    X_input = pd.DataFrame([row])

    # Confidence via predict_proba if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)[0]
        confidence = round(max(proba) * 100, 1)
    else:
        confidence = None

    prediction = model.predict(X_input)[0]
    location = le.inverse_transform([prediction])[0]
    return location, confidence, networks

# ── HEADER ────────────────────────────────────────────────────
st.markdown("## 📍 Path<span style='color:#E53935'>Fi</span>", unsafe_allow_html=True)
st.caption("Intelligent Indoor Navigator — WiFi Fingerprint + ML")
st.divider()

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Live Location",
    "📊 Model Performance",
    "📶 WiFi Signal Map",
    "🗺️ Navigation"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — LIVE LOCATION
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Where am I right now?")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📡 Scan & Predict", use_container_width=True, type="primary"):
            with st.spinner("Scanning WiFi..."):
                try:
                    location, confidence, networks = predict_location()
                    log_location(location)
                    display = LOCATION_LABELS.get(location, location.replace('_', ' ').title())

                    # Confidence check
                    if confidence and confidence < 60:
                        st.warning(f"⚠️ Unknown or unmapped area — closest match: **{display}** ({confidence}% confident)")
                    else:
                        conf_text = f" ({confidence}% confident)" if confidence else ""
                        st.success(f"📍 You are at: **{display}**{conf_text}")

                    # Signal bars
                    st.write("**Visible networks:**")
                    visible = {k: v for k, v in networks.items() if v != -100}
                    for net, rssi in list(visible.items())[:6]:
                        name = net.split('_')[1] if '_' in net else net
                        strength = max(0, min(100, int((rssi + 100) * 2)))
                        st.progress(strength, text=f"{name}: {rssi:.1f} dBm")

                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        # Today's summary in live tab
        summary = get_today_summary()
        if summary:
            st.write("**Today's presence:**")
            for loc, count in summary.items():
                label = LOCATION_LABELS.get(loc, loc.replace('_',' ').title())
                st.write(f"• {label} — {count}")
        else:
            st.info("No location history yet today. Hit Scan to start.")

# ════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("ML Model Comparison")
    try:
        df = pd.read_csv('data/wifi_fingerprints.csv')
        df = df.drop(columns=['timestamp']).fillna(-100)
        le2 = LabelEncoder()
        X = df.drop(columns=['location'])
        y = le2.fit_transform(df['location'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        models = {
            'KNN': KNeighborsClassifier(n_neighbors=2),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }

        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            acc = accuracy_score(y_test, m.predict(X_test))
            results[name] = round(acc * 100, 1)

        fig = px.bar(
            x=list(results.keys()),
            y=list(results.values()),
            labels={'x': 'Model', 'y': 'Accuracy (%)'},
            title='Model Accuracy Comparison',
            color=list(results.values()),
            color_continuous_scale='teal',
            text=list(results.values())
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(
                yaxis_range=[0, 110],
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(200,170,255,0.6)', family='Inter'),
                xaxis=dict(gridcolor='rgba(139,92,246,0.08)', color='rgba(180,140,255,0.5)'),
                yaxis=dict(gridcolor='rgba(139,92,246,0.08)', color='rgba(180,140,255,0.5)'),
            )
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(3)
        for i, (name, acc) in enumerate(results.items()):
            with cols[i]:
                st.metric(label=name, value=f"{acc}%")

        st.caption(f"Dataset: {len(df)} samples • {df['location'].nunique()} locations • 70/30 train-test split")

    except Exception as e:
        st.error(f"Could not load data: {e}")

# ════════════════════════════════════════════════════════════════
# TAB 3 — WIFI SIGNAL MAP
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Signal Strength by Location")
    try:
        df = pd.read_csv('data/wifi_fingerprints.csv')
        df = df.drop(columns=['timestamp']).fillna(-100)
        avg_df = df.groupby('location').mean().reset_index()
        melted = avg_df.melt(id_vars='location', var_name='network', value_name='rssi')
        melted = melted[melted['rssi'] > -100]
        melted['network'] = melted['network'].apply(lambda x: x.split('_')[1] if '_' in x else x)

        fig2 = px.bar(
            melted,
            x='network', y='rssi',
            color='location',
            barmode='group',
            title='Average WiFi Signal Strength per Location',
            labels={'rssi': 'Signal Strength (dBm)', 'network': 'Network'}
        )
        fig2.update_layout(xaxis_tickangle=-45,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='rgba(200,170,255,0.6)', family='Inter'),
                            xaxis=dict(gridcolor='rgba(139,92,246,0.08)', color='rgba(180,140,255,0.5)'),
                            yaxis=dict(gridcolor='rgba(139,92,246,0.08)', color='rgba(180,140,255,0.5)'),
                            legend=dict(
                                bgcolor='rgba(255,255,255,0.03)',
                                bordercolor='rgba(0,212,255,0.2)',
                                borderwidth=1
                                )
                            )
          
           
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Different signal patterns per location — this is what the ML model learns to distinguish.")

    except Exception as e:
        st.error(f"Could not load data: {e}")

# ════════════════════════════════════════════════════════════════
# TAB 4 — NAVIGATION
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Indoor Navigation")

    all_locations = list(LOCATION_GRAPH.keys())
    all_labels = [LOCATION_LABELS.get(l, l) for l in all_locations]

    col1, col2 = st.columns([1, 1])

    with col1:
        # Auto-detect or manual
        use_auto = st.toggle("Auto-detect current location", value=True)

        if use_auto:
            if st.button("📡 Detect My Location", use_container_width=True, type="primary"):
                with st.spinner("Scanning..."):
                    try:
                        location, confidence, _ = predict_location()
                        log_location(location)
                        st.session_state['current_loc'] = location
                        conf_text = f" ({confidence}% confident)" if confidence else ""
                        st.success(f"📍 Detected: **{LOCATION_LABELS.get(location, location)}**{conf_text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            current_loc = st.session_state.get('current_loc', all_locations[0])
        else:
            selected_label = st.selectbox("Select your current location", all_labels)
            current_loc = all_locations[all_labels.index(selected_label)]

        dest_label = st.selectbox(
            "Select destination",
            [l for l in all_labels if all_locations[all_labels.index(l)] != current_loc]
        )
        dest_loc = all_locations[all_labels.index(dest_label)]

        if st.button("🗺️ Find Route", use_container_width=True, type="primary"):
            path, distance = dijkstra(LOCATION_GRAPH, current_loc, dest_loc)

            if path:
                st.session_state['route'] = (path, distance)
            else:
                st.error("No route found between these locations.")

    with col2:
        if 'route' in st.session_state:
            path, distance = st.session_state['route']

            st.write("**Route:**")
            for i, step in enumerate(path):
                label = LOCATION_LABELS.get(step, step.replace('_', ' ').title())
                if i == 0:
                    st.write(f"🟢 **{label}** ← You are here")
                elif i == len(path) - 1:
                    st.write(f"🔴 **{label}** ← Destination")
                else:
                    st.write(f"⬇️ {label}")

            st.divider()
            st.metric("Estimated Distance", f"{distance} m")
            st.metric("Steps", len(path) - 1)
            st.divider()
            st.write("**Step-by-step:**")
            steps = []
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i+1]
                dst_label = LOCATION_LABELS.get(dst, dst)
                d = LOCATION_GRAPH[src][dst]
                direction = DIRECTIONS.get((src, dst), "Walk toward the location")
                steps.append((i+1, direction, d, dst_label))

            for num, direction, d, dst_label in steps:
                st.markdown(
                    f"""<div style="
                        display: flex;
                        align-items: flex-start;
                        gap: 12px;
                        padding: 10px 14px;
                        margin-bottom: 8px;
                        background: rgba(88,28,135,0.12);
                        border: 1px solid rgba(160,100,255,0.12);
                        border-radius: 12px;
                        backdrop-filter: blur(8px);
                    ">
                        <span style="
                            background: linear-gradient(135deg, #10B981, #34D399);
                            color: #022c22;
                            font-size: 0.7rem;
                            font-weight: 700;
                            min-width: 22px;
                            height: 22px;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            flex-shrink: 0;
                            margin-top: 1px;
                            box-shadow: 0 2px 8px rgba(16,185,129,0.4);
                        ">{num}</span>
                        <span style="
                            color: rgba(210,185,255,0.85);
                            font-size: 0.85rem;
                            line-height: 1.5;
                        ">{direction} <span style="color:rgba(255,255,255,0.4)">(~{d}m)</span> <span style="color:rgba(160,100,255,0.7)">→</span> <strong style="color:#EDE9FE">{dst_label}</strong></span>
                    </div>""",
                    unsafe_allow_html=True
                )