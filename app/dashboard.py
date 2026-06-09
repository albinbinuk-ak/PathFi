import streamlit as st
import pandas as pd
import numpy as np
import pickle
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="PathFi",
    page_icon="📍",
    layout="wide"
)

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

# ── HEADER ────────────────────────────────────────────────────
st.title("📍 PathFi")
st.caption("Intelligent Indoor Navigator — WiFi Fingerprint based ML Positioning")
st.divider()

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Live Location", "📊 Model Performance", "📡 WiFi Signal Map"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — LIVE LOCATION
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Where am I right now?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📡 Scan & Predict Location", use_container_width=True, type="primary"):
            with st.spinner("Scanning WiFi networks..."):
                try:
                    model, le, feature_columns = load_model()
                    networks = scan_wifi()

                    # Build feature row matching training columns
                    row = {col: networks.get(col, -100) for col in feature_columns}
                    X_input = pd.DataFrame([row])

                    prediction = model.predict(X_input)[0]
                    location = le.inverse_transform([prediction])[0]

                    st.success(f"📍 You are at: **{location.replace('_', ' ').title()}**")
                    
                    # Show signal strengths
                    st.write("**Detected networks:**")
                    signal_data = {k.split('_')[1]: v for k, v in networks.items() if v != -100}
                    for net, rssi in signal_data.items():
                        strength = int((rssi + 100) * 2)
                        st.progress(strength, text=f"{net}: {rssi:.1f} dBm")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure you have trained the model first.")

    with col2:
        st.info("""
        **How PathFi works:**
        
        1. Scans all visible WiFi networks
        2. Records signal strength (RSSI) of each
        3. Feeds signals into trained ML model
        4. Model matches pattern to known location
        
        *Same technology used in LoRa IoT asset tracking systems*
        """)

# ════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("ML Model Comparison")
    
    try:
        df = pd.read_csv('data/wifi_fingerprints.csv')
        df = df.drop(columns=['timestamp']).fillna(-100)
        
        from sklearn.preprocessing import LabelEncoder
        le_tab2 = LabelEncoder()
        X = df.drop(columns=['location'])
        y = le_tab2.fit_transform(df['location'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=2),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            acc = accuracy_score(y_test, m.predict(X_test))
            results[name] = round(acc * 100, 1)
        
        # Bar chart
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
        fig.update_layout(yaxis_range=[0, 110], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics row
        cols = st.columns(3)
        for i, (name, acc) in enumerate(results.items()):
            with cols[i]:
                st.metric(label=name, value=f"{acc}%")
                
        st.caption(f"Dataset: {len(df)} samples across {df['location'].nunique()} locations")

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
        
        # Average RSSI per location
        avg_df = df.groupby('location').mean().reset_index()
        
        # Melt for plotting
        melted = avg_df.melt(
            id_vars='location',
            var_name='network',
            value_name='rssi'
        )
        melted = melted[melted['rssi'] > -100]
        melted['network'] = melted['network'].apply(lambda x: x.split('_')[1] if '_' in x else x)
        
        fig2 = px.bar(
            melted,
            x='network',
            y='rssi',
            color='location',
            barmode='group',
            title='Average WiFi Signal Strength per Location',
            labels={'rssi': 'Signal Strength (dBm)', 'network': 'Network'}
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.caption("More negative = weaker signal. Patterns differ per location — this is what the ML model learns.")

    except Exception as e:
        st.error(f"Could not load data: {e}")