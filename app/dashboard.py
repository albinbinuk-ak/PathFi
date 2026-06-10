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
st.set_page_config(
    page_title="PathFi",
    page_icon="📍",
    layout="wide"
)


# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0A0F1E 0%, #0D1628 50%, #0A1020 100%);
}

/* ── HEADER ── */
h1 {
    background: linear-gradient(90deg, #00D4FF, #7B6FFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem !important;
    letter-spacing: -0.5px;
}

.stApp > header {
    background: transparent;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(0,212,255,0.1);
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 0.85rem;
    padding: 8px 16px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,111,255,0.15)) !important;
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #7B6FFF);
    color: #0A0F1E;
    font-weight: 700;
    font-size: 0.9rem;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,212,255,0.25);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,212,255,0.4);
    background: linear-gradient(135deg, #33DDFF, #9B8FFF);
}

.stButton > button:active {
    transform: translateY(0px);
}

/* ── CARDS / CONTAINERS ── */
[data-testid="stVerticalBlock"] > div {
    border-radius: 12px;
}

.stAlert {
    border-radius: 10px;
    border: none !important;
}

/* ── SUCCESS / WARNING / ERROR ── */
[data-testid="stAlert"][data-type="success"] {
    background: rgba(0, 212, 100, 0.1) !important;
    border-left: 3px solid #00D464 !important;
    color: #00D464 !important;
}

[data-testid="stAlert"][data-type="warning"] {
    background: rgba(255, 180, 0, 0.1) !important;
    border-left: 3px solid #FFB400 !important;
}

[data-testid="stAlert"][data-type="error"] {
    background: rgba(255, 80, 80, 0.1) !important;
    border-left: 3px solid #FF5050 !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 12px;
    padding: 16px !important;
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(0,212,255,0.3);
    background: rgba(0,212,255,0.05);
}

[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    color: #00D4FF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600;
}

/* ── PROGRESS BARS (signal strength) ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00D4FF, #7B6FFF) !important;
    border-radius: 4px;
}

.stProgress > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 4px;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stSelectbox > div > div:hover {
    border-color: rgba(0,212,255,0.4) !important;
}

/* ── TOGGLE ── */
.stCheckbox, .stToggle {
    color: rgba(255,255,255,0.7);
}

/* ── DIVIDER ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
    margin: 16px 0;
}

/* ── CAPTION ── */
.stCaption {
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.3px;
}

/* ── SUBHEADER ── */
h2, h3 {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 600;
    letter-spacing: -0.3px;
}

/* ── INFO BOX ── */
[data-testid="stAlert"][data-type="info"] {
    background: rgba(123,111,255,0.08) !important;
    border-left: 3px solid #7B6FFF !important;
    border-radius: 10px;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: #00D4FF !important;
}

/* ── SIDEBAR (if used) ── */
[data-testid="stSidebar"] {
    background: rgba(10,15,30,0.95);
    border-right: 1px solid rgba(0,212,255,0.1);
}

/* ── PLOTLY CHARTS background ── */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* ── SIGNAL PULSE ANIMATION for scan button ── */
@keyframes pulse-ring {
    0% { box-shadow: 0 0 0 0 rgba(0,212,255,0.4); }
    70% { box-shadow: 0 0 0 12px rgba(0,212,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
}

.stButton > button[kind="primary"] {
    animation: pulse-ring 2s infinite;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {
    width: 4px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(0,212,255,0.3);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0,212,255,0.6);
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
st.title("📍 PathFi")
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
                font=dict(color='rgba(255,255,255,0.7)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
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
                            font=dict(color='rgba(255,255,255,0.7)'),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
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

            # Step by step
            st.write("**Step-by-step:**")
            # Step by step
            st.write("**Step-by-step:**")
            steps = []
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i+1]
                dst_label = LOCATION_LABELS.get(dst, dst)
                d = LOCATION_GRAPH[src][dst]
                direction = DIRECTIONS.get((src, dst), "Walk toward the location")
                steps.append(f"{i+1}. {direction} (~{d}m) → **{dst_label}**")
            
            for step in steps:
                st.write(step)