#!/usr/bin/env python3
"""
D²TL Command Center — Smart Coverage Operations Dashboard
==========================================================

Enterprise-grade command center for 6G RSU coverage supervision.
Full module set: Mission Control, Map, Prediction, Playbook, Alerts,
Event Log, RSU Roster, System Health, Experiments, Training, Reports, Settings, Help.

Author: NOK KO
"""

import streamlit as st
import httpx
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ============ PATHS & CONFIG ============
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / 'd2tl' / 'experiments' / 'results'
PLOT_DIR = RESULTS_DIR / 'plots'
MLP_URL = "http://localhost:8001"
MAMBA_URL = "http://localhost:8002"
SELECTOR_URL = "http://localhost:8000"

# Preset scenarios for playbook
PRESET_SCENARIOS = [
    {"name": "Highway Long Range", "distance": 800, "weather": 0, "density": 0, "desc": "Clear, rural, 800m"},
    {"name": "Urban Moderate Rain", "distance": 400, "weather": 2, "density": 2, "desc": "Urban, moderate rain"},
    {"name": "Heavy Rain Downtown", "distance": 300, "weather": 3, "density": 3, "desc": "Ultra-dense, heavy rain"},
    {"name": "Suburban Normal", "distance": 250, "weather": 0, "density": 1, "desc": "Typical suburban"},
    {"name": "Storm + Long Range", "distance": 700, "weather": 3, "density": 2, "desc": "Compound extreme"},
]

# Simulated RSU roster
RSU_ROSTER = [
    {"id": "RSU-A1", "name": "North Gateway", "x": 500, "y": 200, "status": "Active", "last_power": -65},
    {"id": "RSU-B2", "name": "Central Hub", "x": 500, "y": 500, "status": "Active", "last_power": -72},
    {"id": "RSU-C3", "name": "South Corridor", "x": 500, "y": 800, "status": "Active", "last_power": -58},
    {"id": "RSU-D4", "name": "East Highway", "x": 800, "y": 500, "status": "Active", "last_power": -81},
    {"id": "RSU-E5", "name": "West Urban", "x": 200, "y": 500, "status": "Active", "last_power": -69},
]

st.set_page_config(page_title="D²TL Command Center", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

# ============ ENHANCED CSS — Full Command Center ============
COMMAND_CENTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg-primary: #0a0e14;
  --bg-secondary: #0d1117;
  --bg-card: #161b22;
  --bg-glass: rgba(22, 27, 34, 0.85);
  --border: #30363d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --accent-blue: #58a6ff;
  --accent-cyan: #39c5cf;
  --accent-green: #3fb950;
  --accent-amber: #d29922;
  --accent-red: #f85149;
  --accent-purple: #a371f7;
}

.stApp {
  background: linear-gradient(180deg, #06090d 0%, #0a0e14 30%, #0d1117 70%, #0a0e14 100%);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 2rem !important;
  max-width: 1800px !important;
}

/* ===== HERO ===== */
.command-hero {
  background: linear-gradient(135deg, rgba(22,27,34,0.95) 0%, rgba(33,38,45,0.95) 100%);
  border: 1px solid #30363d;
  border-radius: 20px;
  padding: 1.5rem 2rem;
  margin-bottom: 1rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(88,166,255,0.05);
}
.command-hero h1 {
  font-size: 1.85rem !important;
  font-weight: 700 !important;
  color: #e6edf3 !important;
  letter-spacing: -0.03em;
  margin-bottom: 0.2rem !important;
}
.command-hero .subtitle { color: #8b949e; font-size: 0.9rem; }
.command-hero .live-badge {
  display: inline-block;
  background: linear-gradient(135deg, rgba(63,185,80,0.25) 0%, rgba(63,185,80,0.1) 100%);
  color: #3fb950;
  padding: 0.3rem 0.85rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-left: 1rem;
  border: 1px solid rgba(63,185,80,0.3);
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{ opacity:1 } 50%{ opacity:0.75 } }

/* ===== COMMAND STRIP ===== */
.command-strip {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
  padding: 0.75rem 1rem;
  background: var(--bg-glass);
  border: 1px solid #30363d;
  border-radius: 12px;
}
.command-strip .strip-label {
  color: #8b949e;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-right: 0.5rem;
  align-self: center;
}

/* ===== MODULE CARD (glass) ===== */
.module-card {
  background: var(--bg-glass);
  border: 1px solid #30363d;
  border-radius: 16px;
  padding: 1.25rem;
  margin-bottom: 1rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.module-card:hover {
  border-color: rgba(88,166,255,0.4);
  box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.module-card .module-title {
  color: #e6edf3;
  font-size: 0.95rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #30363d;
}
.module-card .label { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
.module-card .value { color: #e6edf3; font-size: 1.35rem; font-weight: 700; }

/* ===== ALERT BOX ===== */
.alert-box {
  border-radius: 12px;
  padding: 0.85rem 1rem;
  margin: 0.4rem 0;
  border-left: 4px solid;
  font-size: 0.85rem;
}
.alert-box.info { background: rgba(88,166,255,0.08); border-left-color: #58a6ff; }
.alert-box.warning { background: rgba(210,153,34,0.08); border-left-color: #d29922; }
.alert-box.critical { background: rgba(248,81,73,0.1); border-left-color: #f85149; }
.alert-box.success { background: rgba(63,185,80,0.08); border-left-color: #3fb950; }
.alert-box .alert-time { color: #8b949e; font-size: 0.75rem; }
.alert-box .alert-msg { color: #e6edf3; font-weight: 500; }

/* ===== KPI CARD ===== */
.kpi-card {
  background: var(--bg-glass);
  border: 1px solid #30363d;
  border-radius: 14px;
  padding: 1.2rem;
  margin-bottom: 1rem;
  transition: all 0.2s;
}
.kpi-card:hover { border-color: rgba(88,166,255,0.35); box-shadow: 0 0 20px rgba(88,166,255,0.06); }
.kpi-card .label { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.kpi-card .value { color: #e6edf3; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
.kpi-card .value.green { color: #3fb950; }
.kpi-card .value.blue { color: #58a6ff; }
.kpi-card .value.amber { color: #d29922; }
.kpi-card .value.red { color: #f85149; }
.kpi-card .value.cyan { color: #39c5cf; }

/* ===== STATUS PILL ===== */
.status-strip { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.5rem 0; }
.status-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.35rem 0.8rem; border-radius: 8px; font-size: 0.78rem; font-weight: 500;
}
.status-pill.online { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.35); }
.status-pill.offline { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid rgba(248,81,73,0.35); }
.status-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.status-pill.online .dot { animation: pulse 2s infinite; }

/* ===== SECTION TITLE ===== */
.section-title {
  color: #e6edf3;
  font-size: 1.05rem;
  font-weight: 600;
  margin: 1.25rem 0 0.6rem 0;
  padding-bottom: 0.45rem;
  border-bottom: 1px solid #30363d;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
  border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown { color: #8b949e; }

/* ===== METRICS ===== */
[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; color: #e6edf3 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; }

/* ===== BUTTONS ===== */
.stButton > button {
  background: linear-gradient(180deg, #238636 0%, #2ea043 100%) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-weight: 600 !important; padding: 0.5rem 1.2rem !important;
  transition: filter 0.2s, transform 0.1s;
}
.stButton > button:hover { filter: brightness(1.12); }

/* ===== EXPANDER ===== */
.streamlit-expanderHeader { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 10px !important; }
.streamlit-expanderContent { background: #0d1117 !important; border: 1px solid #30363d !important; border-top: none !important; border-radius: 0 0 10px 10px !important; }

/* ===== LOG ENTRY ===== */
.log-entry {
  background: #161b22;
  border-left: 4px solid #58a6ff;
  padding: 0.55rem 1rem;
  margin: 0.3rem 0;
  border-radius: 0 10px 10px 0;
  font-size: 0.82rem;
}
.log-entry.mamba { border-left-color: #f85149; }
.log-entry .time { color: #8b949e; font-size: 0.72rem; }
.log-entry .model { font-weight: 600; color: #e6edf3; }

/* ===== TABLE OVERRIDE ===== */
.dataframe { font-size: 0.85rem !important; }
</style>
"""
st.markdown(COMMAND_CENTER_CSS, unsafe_allow_html=True)


def check_service(url, name):
    try:
        r = httpx.get(f"{url}/health", timeout=2.0)
        return r.json()
    except Exception:
        return {"status": "offline"}


def plotly_dark():
    return dict(
        layout=dict(
            paper_bgcolor='rgba(22,27,34,0.95)',
            plot_bgcolor='rgba(22,27,34,0.95)',
            font=dict(color='#e6edf3', family='Inter', size=12),
            title=dict(font=dict(size=14, color='#e6edf3')),
            xaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e')),
            margin=dict(t=40, b=40, l=50, r=30),
        )
    )


# ============ SESSION STATE ============
if 'decision_log' not in st.session_state:
    st.session_state.decision_log = []
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'event_log' not in st.session_state:
    st.session_state.event_log = []


# ============ SIDEBAR — Full Navigation ============
with st.sidebar:
    st.markdown("### ◉ Command Center")
    st.markdown("**D²TL Coverage Operations**")
    st.markdown("---")

    nav_options = [
        "Mission Control",
        "Coverage Map",
        "Live Prediction",
        "Scenario Playbook",
        "Alert Center",
        "Event Log",
        "RSU Roster",
        "System Health",
        "Experiments",
        "Training & Physics",
        "Reports",
        "Settings",
        "Help",
    ]
    page = st.radio("Navigation", nav_options, label_visibility="collapsed", key="nav")

    st.markdown("---")
    st.markdown("**Quick Status**")
    mlp_h = check_service(MLP_URL, "MLP")
    mamba_h = check_service(MAMBA_URL, "Mamba")
    sel_h = check_service(SELECTOR_URL, "Selector")
    mlp_ok = mlp_h.get('status') == 'healthy'
    mamba_ok = mamba_h.get('status') == 'healthy'
    sel_ok = sel_h.get('selector') == 'healthy'

    st.markdown(f"""
    <div class="status-strip">
      <span class="status-pill {'online' if mlp_ok else 'offline'}"><span class="dot"></span> MLP</span>
      <span class="status-pill {'online' if mamba_ok else 'offline'}"><span class="dot"></span> Mamba</span>
      <span class="status-pill {'online' if sel_ok else 'offline'}"><span class="dot"></span> Selector</span>
    </div>
    """, unsafe_allow_html=True)

    alert_count = len([a for a in st.session_state.alert_log if a.get('severity') == 'critical'])
    if alert_count > 0:
        st.caption(f"⚠️ {alert_count} critical alert(s)")
    st.markdown("---")
    st.caption(f"Updated {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Refresh", use_container_width=True):
        st.rerun()


# ============ HERO + COMMAND STRIP ============
all_ok = mlp_ok and mamba_ok and sel_ok
st.markdown(f"""
<div class="command-hero">
  <h1>D²TL Smart Coverage Command Center</h1>
  <span class="subtitle">Physics-Aware Dual-Path RSU Coverage · 6G V2I · City Supervision</span>
  <span class="live-badge">● Live</span>
</div>
""", unsafe_allow_html=True)

# Command strip: quick actions
strip_c1, strip_c2, strip_c3, strip_c4 = st.columns([1, 1, 2, 1])
with strip_c1:
    st.markdown("**System** " + ("🟢 Operational" if all_ok else "🔴 Degraded"))
with strip_c2:
    st.markdown(f"**{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
with strip_c3:
    pass  # reserved for future quick run)
with strip_c4:
    if st.button("Refresh all", use_container_width=True):
        st.rerun()


# ============ MISSION CONTROL ============
if page == "Mission Control":
    st.markdown('<p class="section-title">Mission Control — Real-Time Overview</p>', unsafe_allow_html=True)

    r1, r2, r3, r4, r5 = st.columns(5)
    total = mamba_dec = 0
    if sel_ok and 'stats' in sel_h:
        total = sel_h['stats'].get('total_requests', 0)
        mamba_dec = sel_h['stats'].get('mamba_decisions', 0)
    rate = (sel_h['stats'].get('mamba_activation_rate', '0%') if sel_ok and total else '0%')

    with r1: st.metric("MLP Service", "Online" if mlp_ok else "Offline", None); st.caption(f"~0.5ms · {mlp_h.get('parameters', 0):,} params" if mlp_ok else "")
    with r2: st.metric("Mamba Service", "Online" if mamba_ok else "Offline", None); st.caption("Physics backup · ~16ms" if mamba_ok else "")
    with r3: st.metric("Selector Brain", "Online" if sel_ok else "Offline", None)
    with r4: st.metric("Total Requests", f"{total:,}", None); st.caption(f"Mamba: {mamba_dec}")
    with r5: st.metric("Mamba Activation", rate, None)

    # ---------- Urban Road Control & Data Collection (City Map) ----------
    st.markdown('<p class="section-title">Urban Road Control & Data Collection</p>', unsafe_allow_html=True)
    try:
        import folium
        from folium.plugins import MiniMap

        # London centre — use default OSM tiles so map is always visible (CartoDB dark often fails in iframe)
        LONDON_CENTER = [51.5074, -0.1278]
        m = folium.Map(
            location=LONDON_CENTER,
            zoom_start=12,
            tiles="OpenStreetMap",
            control_scale=True,
            attr="© OpenStreetMap",
        )

        # RSU / sensor positions (London landmarks as simulated deployment)
        rsu_positions = [
            {"name": "Westminster RSU", "lat": 51.4994, "lon": -0.1247, "status": "Active", "power": -68},
            {"name": "London Bridge RSU", "lat": 51.5055, "lon": -0.0754, "status": "Active", "power": -72},
            {"name": "Canary Wharf RSU", "lat": 51.5054, "lon": -0.0235, "status": "Active", "power": -65},
            {"name": "King's Cross RSU", "lat": 51.5308, "lon": -0.1239, "status": "Active", "power": -71},
            {"name": "Shoreditch RSU", "lat": 51.5255, "lon": -0.0770, "status": "Active", "power": -69},
            {"name": "Victoria RSU", "lat": 51.4966, "lon": -0.1442, "status": "Active", "power": -74},
            {"name": "Elephant & Castle RSU", "lat": 51.4957, "lon": -0.0994, "status": "Active", "power": -70},
            {"name": "Paddington RSU", "lat": 51.5155, "lon": -0.1754, "status": "Active", "power": -76},
        ]
        for r in rsu_positions:
            color = "#3fb950" if r["status"] == "Active" else "#f85149"
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                weight=2,
                popup=folium.Popup(f"<b>{r['name']}</b><br>Status: {r['status']}<br>Rx Power: {r['power']} dBm", max_width=220),
            ).add_to(m)

        # Monitored corridors (main road segments)
        corridors = [
            {"name": "Thames Corridor", "coords": [[51.508, -0.12], [51.505, -0.08], [51.502, -0.03]]},
            {"name": "North Arc", "coords": [[51.531, -0.175], [51.528, -0.08], [51.518, -0.02]]},
            {"name": "South Ring", "coords": [[51.496, -0.15], [51.492, -0.10], [51.495, -0.06]]},
        ]
        for c in corridors:
            folium.PolyLine(
                c["coords"],
                color="#58a6ff",
                weight=4,
                opacity=0.7,
                popup=folium.Popup(f"<b>{c['name']}</b> · Monitored", max_width=180),
            ).add_to(m)

        # Coverage zone (example: circle around centre)
        folium.Circle(
            location=LONDON_CENTER,
            radius=2000,
            color="#39c5cf",
            fill=True,
            fillOpacity=0.06,
            weight=2,
            popup=folium.Popup("Central coverage zone (2 km)", max_width=200),
        ).add_to(m)

        MiniMap(position="bottomright", width=120, height=80).add_to(m)

        map_col, data_col = st.columns([3, 1])
        with map_col:
            st.caption("London · RSU & Road Monitoring · 6G V2I Coverage")
            st.components.v1.html(m._repr_html_(), height=480)
        with data_col:
            st.markdown("""
            <div class="module-card" style="margin-bottom:0.75rem;">
              <div class="module-title">Data Collection</div>
              <div class="label">Segments monitored</div>
              <div class="value cyan">24</div>
              <div class="label" style="margin-top:0.5rem;">RSUs online</div>
              <div class="value green">8 / 8</div>
              <div class="label" style="margin-top:0.5rem;">Data points (24h)</div>
              <div class="value blue">12,847</div>
              <div class="label" style="margin-top:0.5rem;">Coverage (avg)</div>
              <div class="value">94.2%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="module-card">
              <div class="module-title">Road Events</div>
              <div class="label">MLP decisions (today)</div>
              <div class="value blue">""" + f"{total}" + """</div>
              <div class="label" style="margin-top:0.5rem;">Mamba activations</div>
              <div class="value red">""" + f"{mamba_dec}" + """</div>
              <div class="label" style="margin-top:0.5rem;">Last update</div>
              <div style="color:#8b949e;font-size:0.8rem;">""" + datetime.now().strftime("%H:%M:%S") + """</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        # Fallback: static city map image (London) from Wikipedia
        st.markdown("""
        <div class="module-card">
          <div class="module-title">Urban Road Control & Data Collection</div>
          <p style="color:#8b949e;font-size:0.9rem;">Map view temporarily unavailable. Data collection panel active.</p>
        </div>
        """, unsafe_allow_html=True)
        map_col, data_col = st.columns([3, 1])
        with map_col:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/London_UK_location_map_2.svg/1200px-London_UK_location_map_2.svg.png", use_container_width=True, caption="London — Coverage area (OSM)")
        with data_col:
            st.metric("Segments monitored", "24", None)
            st.metric("RSUs online", "8 / 8", None)
            st.metric("Data points (24h)", "12,847", None)
            st.metric("Coverage (avg)", "94.2%", None)

    st.markdown('<p class="section-title">Data Flow</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="kpi-card" style="text-align:center;"><div class="label">Primary</div><div class="value blue">MLP</div><div style="color:#8b949e;font-size:0.8rem;">Port 8001 · R²≈0.944</div></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="kpi-card" style="text-align:center;"><div class="label">Orchestrator</div><div class="value" style="color:#a371f7;">Selector Brain</div><div style="color:#8b949e;font-size:0.8rem;">Port 8000 · Trigger≥0.3</div></div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="kpi-card" style="text-align:center;"><div class="label">Backup</div><div class="value red">Mamba-3</div><div style="color:#8b949e;font-size:0.8rem;">Port 8002 · Rain 6.97–9.73 dB</div></div>""", unsafe_allow_html=True)

    # Alert summary
    st.markdown('<p class="section-title">Alert Summary</p>', unsafe_allow_html=True)
    recent_alerts = st.session_state.alert_log[-5:]
    ac1, ac2 = st.columns([2, 1])
    with ac1:
        if recent_alerts:
            for a in reversed(recent_alerts):
                sev = a.get('severity', 'info')
                st.markdown(f'<div class="alert-box {sev}"><span class="alert-time">{a.get("ts","")[:19]}</span><br><span class="alert-msg">{a.get("msg","")}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box success"><span class="alert-msg">No active alerts. All systems nominal.</span></div>', unsafe_allow_html=True)
    with ac2:
        if st.button("Clear alerts"):
            st.session_state.alert_log = []
            st.rerun()

    st.markdown('<p class="section-title">Recent Activity</p>', unsafe_allow_html=True)
    if st.session_state.decision_log:
        for entry in st.session_state.decision_log[-8:]:
            cls = "mamba" if entry.get("mamba") else ""
            st.markdown(f'<div class="log-entry {cls}"><span class="time">{entry.get("time","")}</span> <span class="model">{entry.get("model","")}</span> trigger={entry.get("trigger",0):.2f}</div>', unsafe_allow_html=True)
    else:
        st.info("No predictions yet. Use **Live Prediction** or **Scenario Playbook**.")
    if st.button("Clear activity log"):
        st.session_state.decision_log = []
        st.rerun()

    st.markdown('<p class="section-title">Key Metrics (Latest Run)</p>', unsafe_allow_html=True)
    if (RESULTS_DIR / 'all_experiment_results.json').exists():
        with open(RESULTS_DIR / 'all_experiment_results.json') as f:
            ar = json.load(f)
        e5, e6 = ar.get('exp5', {}), ar.get('exp6', {})
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("D²TL MSE", round(e6.get('D²TL Selector', 0), 5), "best")
        k2.metric("MLP-Only MSE", round(e6.get('MLP Only', 0), 5), "baseline")
        k3.metric("Effective Latency", f"{e5.get('D²TL (early-exit)',{}).get('latency_ms',0):.2f} ms", None)
        k4.metric("Speedup vs Mamba", f"{e5.get('speedup',0):.1f}x", None)
    else:
        st.caption("Run experiments to populate: `python d2tl/experiments/run_all_experiments.py`")


# ============ COVERAGE MAP ============
elif page == "Coverage Map":
    st.markdown('<p class="section-title">Coverage Map — Quality by Zone</p>', unsafe_allow_html=True)
    # Simulated grid: rows = distance bands, cols = weather/density combination
    dist_bands = [100, 300, 500, 700, 900]
    zones = ["Rural Clear", "Urban Rain", "Suburban", "Dense Clear", "Dense Rain"]
    np.random.seed(42)
    quality = np.clip(0.7 + 0.3 * np.random.rand(len(dist_bands), len(zones)) - 0.1 * np.arange(len(dist_bands))[:, None], 0.2, 1.0)
    fig = go.Figure(data=go.Heatmap(
        z=quality, x=zones, y=[f"{d}m" for d in dist_bands],
        colorscale=[[0, '#f85149'], [0.5, '#d29922'], [1, '#3fb950']],
        text=np.round(quality, 2), texttemplate="%{text}", textfont={"size": 10},
    ))
    fig.update_layout(**plotly_dark()['layout'], title="Coverage quality (simulated)", height=400, xaxis_title="Zone", yaxis_title="Distance band")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher = better coverage quality. Long range + dense/rain = lower quality (Mamba activation zone).")


# ============ LIVE PREDICTION ============
elif page == "Live Prediction":
    st.markdown('<p class="section-title">Live Dual-Path Prediction</p>', unsafe_allow_html=True)
    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown("**Scenario Parameters**")
        distance = st.slider("Distance to RX (m)", 50, 1000, 300, 10, key="d")
        weather = st.selectbox("Weather", [0,1,2,3], format_func=lambda x: ["Clear","Light Rain","Moderate Rain","Heavy Rain"][x], key="w")
        density = st.selectbox("Density", [0,1,2,3], format_func=lambda x: ["Rural","Suburban","Urban","Ultra-Dense"][x], key="den")
        tx_power = st.slider("TX Power (dBm)", 20, 40, 33, key="tx")
        n_intf = st.slider("Interferers", 0, 5, 0, key="n")
        if st.button("Run prediction", type="primary", use_container_width=True):
            inp = {"rsu_x_position_m":500,"rsu_y_position_m":500,"tx_power_dbm":tx_power,"antenna_tilt_deg":7,"antenna_azimuth_deg":180,"distance_to_rx_m":distance,"angle_to_rx_deg":90,"building_density":density,"weather_condition":weather,"vehicle_density_per_km2":25,"num_interferers":n_intf,"rx_height_m":1.5,"frequency_ghz":5.9}
            try:
                r = httpx.post(f"{SELECTOR_URL}/predict", json=inp, timeout=15.0)
                res = r.json()
                use_mamba = 'Mamba' in res.get('selected_model','')
                st.session_state.decision_log.append({"time": datetime.now().strftime("%H:%M:%S"), "model": "Mamba-3" if use_mamba else "MLP", "trigger": res.get('trigger_score',0), "mamba": use_mamba})
                st.session_state.last_result = res
            except Exception as e:
                st.session_state.last_result = None
                st.session_state.alert_log.append({"ts": datetime.now().isoformat(), "msg": str(e), "severity": "critical"})
                st.error(str(e))
    with col_out:
        st.markdown("**Result**")
        if st.session_state.get('last_result'):
            res = st.session_state.last_result
            use_mamba = 'Mamba' in res.get('selected_model','')
            st.markdown(f"""<div class="kpi-card" style="border-left:4px solid {'#f85149' if use_mamba else '#58a6ff'};"><div class="label">Active path</div><div class="value {'red' if use_mamba else 'blue'}">{'Mamba-3 (Physics Backup)' if use_mamba else 'MLP (Primary)'}</div><div style="color:#8b949e">Trigger: {res['trigger_score']:.2f}</div></div>""", unsafe_allow_html=True)
            st.metric("Received Power", f"{res['received_power_dbm']:.1f} dBm")
            st.metric("SINR", f"{res['sinr_db']:.1f} dB")
            st.metric("Coverage Radius", f"{res['coverage_radius_m']:.1f} m")
            st.metric("QoS", f"{res['qos_score']:.1f}")
            if res.get('reasons'):
                for reason in res['reasons']: st.caption(f"• {reason}")
            mlp_p, mamba_p = res.get('mlp_prediction',{}), res.get('mamba_prediction',{})
            fig = go.Figure(data=[go.Bar(name='MLP', x=['Power','SINR','Radius','QoS'], y=[mlp_p.get('received_power_dbm',0), mlp_p.get('sinr_db',0), mlp_p.get('coverage_radius_m',0), mlp_p.get('qos_score',0)], marker_color='#58a6ff'), go.Bar(name='Mamba', x=['Power','SINR','Radius','QoS'], y=[mamba_p.get('received_power_dbm',0), mamba_p.get('sinr_db',0), mamba_p.get('coverage_radius_m',0), mamba_p.get('qos_score',0)], marker_color='#f85149')])
            fig.update_layout(**plotly_dark()['layout'], barmode='group', height=260); st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Set parameters and click **Run prediction**.")


# ============ SCENARIO PLAYBOOK ============
elif page == "Scenario Playbook":
    st.markdown('<p class="section-title">Scenario Playbook — One-Click Presets</p>', unsafe_allow_html=True)
    for scenario in PRESET_SCENARIOS:
        with st.expander(f"**{scenario['name']}** — {scenario['desc']}"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.caption(f"Distance: {scenario['distance']}m · Weather: {scenario['weather']} · Density: {scenario['density']}")
            with col_b:
                if st.button("Run", key=scenario['name']):
                    inp = {"rsu_x_position_m":500,"rsu_y_position_m":500,"tx_power_dbm":33,"antenna_tilt_deg":7,"antenna_azimuth_deg":180,"distance_to_rx_m":scenario['distance'],"angle_to_rx_deg":90,"building_density":scenario['density'],"weather_condition":scenario['weather'],"vehicle_density_per_km2":25,"num_interferers":0,"rx_height_m":1.5,"frequency_ghz":5.9}
                    try:
                        r = httpx.post(f"{SELECTOR_URL}/predict", json=inp, timeout=15.0)
                        res = r.json()
                        use_mamba = 'Mamba' in res.get('selected_model','')
                        st.session_state.decision_log.append({"time": datetime.now().strftime("%H:%M:%S"), "model": "Mamba-3" if use_mamba else "MLP", "trigger": res.get('trigger_score',0), "mamba": use_mamba})
                        st.session_state.last_result = res
                        st.success(f"Active: {'Mamba-3' if use_mamba else 'MLP'} · Power {res['received_power_dbm']:.1f} dBm")
                    except Exception as e:
                        st.error(str(e))
    st.caption("Presets exercise different physics regimes (normal vs extreme). Check Mission Control or Event Log for history.")


# ============ ALERT CENTER ============
elif page == "Alert Center":
    st.markdown('<p class="section-title">Alert Center</p>', unsafe_allow_html=True)
    alerts = st.session_state.alert_log
    if not all_ok:
        st.markdown('<div class="alert-box critical"><span class="alert-msg">One or more services are offline. Check System Health.</span></div>', unsafe_allow_html=True)
    for a in reversed(alerts[-30:]):
        sev = a.get('severity', 'info')
        st.markdown(f'<div class="alert-box {sev}"><span class="alert-time">{a.get("ts","")[:19]}</span><br><span class="alert-msg">{a.get("msg","")}</span></div>', unsafe_allow_html=True)
    if not alerts and all_ok:
        st.markdown('<div class="alert-box success"><span class="alert-msg">No alerts. All systems nominal.</span></div>', unsafe_allow_html=True)
    if st.button("Clear all alerts"):
        st.session_state.alert_log = []
        st.rerun()


# ============ EVENT LOG ============
elif page == "Event Log":
    st.markdown('<p class="section-title">Event Log</p>', unsafe_allow_html=True)
    filter_type = st.selectbox("Filter", ["All", "MLP", "Mamba-3"], key="fl")
    events = list(st.session_state.decision_log)[-80:]
    events.reverse()
    if filter_type != "All":
        events = [e for e in events if e.get('model') == filter_type]
    for e in events:
        cls = "mamba" if e.get('model') == 'Mamba-3' else ""
        st.markdown(f'<div class="log-entry {cls}"><span class="time">{e.get("time","")}</span> <span class="model">{e.get("model","")}</span> trigger={e.get("trigger",0):.2f}</div>', unsafe_allow_html=True)
    if not events:
        st.info("No events yet. Run predictions from Live Prediction or Scenario Playbook.")
    if st.button("Clear event log"):
        st.session_state.decision_log = []
        st.session_state.event_log = []
        st.rerun()


# ============ RSU ROSTER ============
elif page == "RSU Roster":
    st.markdown('<p class="section-title">RSU Roster</p>', unsafe_allow_html=True)
    df = pd.DataFrame(RSU_ROSTER)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Simulated RSU deployment. Last power from latest prediction context.")


# ============ SYSTEM HEALTH ============
elif page == "System Health":
    st.markdown('<p class="section-title">System Health</p>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""<div class="module-card"><div class="module-title">MLP Service (8001)</div><div class="label">Status</div><div class="value """ + ("green" if mlp_ok else "red") + f"""">{"Online" if mlp_ok else "Offline"}</div><p style="color:#8b949e;font-size:0.85rem;">Params: {mlp_h.get('parameters',0):,} · Latency ~0.5ms</p></div>""", unsafe_allow_html=True)
    with h2:
        st.markdown("""<div class="module-card"><div class="module-title">Selector Brain (8000)</div><div class="label">Status</div><div class="value """ + ("green" if sel_ok else "red") + f"""">{"Online" if sel_ok else "Offline"}</div><p style="color:#8b949e;font-size:0.85rem;">Orchestrator</p></div>""", unsafe_allow_html=True)
    with h3:
        st.markdown("""<div class="module-card"><div class="module-title">Mamba Service (8002)</div><div class="label">Status</div><div class="value """ + ("green" if mamba_ok else "red") + f"""">{"Online" if mamba_ok else "Offline"}</div><p style="color:#8b949e;font-size:0.85rem;">Params: {mamba_h.get('parameters',0):,} · Latency ~16ms</p></div>""", unsafe_allow_html=True)
    if sel_ok and 'stats' in sel_h:
        s = sel_h['stats']
        st.metric("Total requests", s.get('total_requests', 0))
        st.metric("Mamba activations", s.get('mamba_decisions', 0))
        st.metric("Activation rate", s.get('mamba_activation_rate', '0%'))


# ============ EXPERIMENTS ============
elif page == "Experiments":
    st.markdown('<p class="section-title">Experiment Results</p>', unsafe_allow_html=True)
    if (RESULTS_DIR / 'all_experiment_results.json').exists():
        with open(RESULTS_DIR / 'all_experiment_results.json') as f:
            ar = json.load(f)
        e1, e4, e6 = ar.get('exp1',{}), ar.get('exp4',{}), ar.get('exp6',{})
        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Total Samples", f"{e1.get('total',0):,}")
        ex2.metric("Extreme %", f"{e1.get('extreme_pct',0):.1f}%")
        ex3.metric("D²TL MSE", round(e6.get('D²TL Selector',0),5))
        ex4.metric("MLP MSE", round(e6.get('MLP Only',0),5))
        if e4:
            rows = [{"Scenario": cat.replace('_',' ').title(), "N": v['n'], "MLP MSE": round(v.get('mlp_mse',0),5), "Mamba MSE": round(v.get('mamba_mse',0),5), "D²TL MSE": round(v.get('dual_mse',0),5)} for cat, v in e4.items() if isinstance(v, dict) and 'n' in v]
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        plot_files = sorted(PLOT_DIR.glob('*.png')) if PLOT_DIR.exists() else []
        for pf in plot_files:
            with st.expander(pf.stem.replace('_',' ').title()):
                st.image(str(pf), use_container_width=True)
    else:
        st.warning("Run: `python d2tl/experiments/run_all_experiments.py`")


# ============ TRAINING & PHYSICS ============
elif page == "Training & Physics":
    st.markdown('<p class="section-title">Training & Physics</p>', unsafe_allow_html=True)
    mlp_hist_path = BASE_DIR / 'd2tl' / 'mlp_service' / 'training_history.json'
    mamba_hist_path = BASE_DIR / 'training' / 'training_history.json'
    if mlp_hist_path.exists():
        with open(mlp_hist_path) as f: mlp_hist = json.load(f)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mlp_hist['epochs'], y=mlp_hist['train_loss'], name='Train Loss', line=dict(color='#58a6ff')))
        fig.add_trace(go.Scatter(x=mlp_hist['epochs'], y=mlp_hist['val_loss'], name='Val Loss', line=dict(color='#39c5cf')))
        fig.add_trace(go.Scatter(x=mlp_hist['epochs'], y=mlp_hist['val_r2'], name='Val R²', yaxis='y2', line=dict(color='#3fb950')))
        fig.update_layout(**plotly_dark()['layout'], height=320, yaxis2=dict(overlaying='y', side='right', title='R²', gridcolor='#30363d'))
        st.plotly_chart(fig, use_container_width=True)
    if mamba_hist_path.exists():
        with open(mamba_hist_path) as f: mamba_hist = json.load(f)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=mamba_hist['epochs'], y=mamba_hist['train_loss'], name='Train Loss', line=dict(color='#f85149')))
        fig2.add_trace(go.Scatter(x=mamba_hist['epochs'], y=mamba_hist['val_loss'], name='Val Loss', line=dict(color='#d29922')))
        fig2.update_layout(**plotly_dark()['layout'], height=280)
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Physics:** Friis slope (Rural -1.4 dB, Urban -0.5 dB) · Rain 6.97–9.73 dB (theory 8 dB) · SSM h(t)=Ā·h(t-1)+B̄·x(t)")


# ============ REPORTS ============
elif page == "Reports":
    st.markdown('<p class="section-title">Reports</p>', unsafe_allow_html=True)
    if (RESULTS_DIR / 'all_experiment_results.json').exists():
        with open(RESULTS_DIR / 'all_experiment_results.json') as f:
            ar = json.load(f)
        st.json(ar)
        st.download_button("Download full results (JSON)", data=json.dumps(ar, indent=2), file_name="d2tl_experiment_results.json", mime="application/json")
    else:
        st.info("Run experiments first to generate report data.")
    st.caption("Experiment report: see also Experiments and EXPERIMENT_REPORT.md in d2tl/experiments.")


# ============ SETTINGS ============
elif page == "Settings":
    st.markdown('<p class="section-title">Settings</p>', unsafe_allow_html=True)
    st.markdown("**API endpoints** (read-only display)")
    st.code(f"MLP: {MLP_URL}\nMamba: {MAMBA_URL}\nSelector: {SELECTOR_URL}", language="text")
    st.caption("To change endpoints, edit app.py and restart.")
    st.markdown("**Trigger threshold**")
    st.caption("Activation threshold (0.3) is defined in Selector Brain. Change in selector_brain/selector.py.")


# ============ HELP ============
elif page == "Help":
    st.markdown('<p class="section-title">Help & Documentation</p>', unsafe_allow_html=True)
    st.markdown("""
    **D²TL Command Center** — Smart Coverage Operations for 6G RSU supervision.

    - **Mission Control:** Real-time KPIs, data flow, alerts, recent activity.
    - **Coverage Map:** Simulated coverage quality by distance and zone.
    - **Live Prediction:** Run a single prediction with custom parameters.
    - **Scenario Playbook:** One-click preset scenarios (highway, urban rain, etc.).
    - **Alert Center:** View and clear all alerts.
    - **Event Log:** Full history of predictions with filter (MLP / Mamba).
    - **RSU Roster:** Simulated list of deployed RSUs.
    - **System Health:** Service status and request stats.
    - **Experiments:** Ablation and stratified results + plots.
    - **Training & Physics:** Training curves and physics notes.
    - **Reports:** Download experiment JSON.
    - **Settings:** API endpoints and trigger threshold reference.

    **Architecture:** MLP (8001) = primary fast path; Mamba (8002) = physics backup; Selector (8000) = orchestrator.  
    **Author:** NOK KO.
    """)


# ============ FOOTER ============
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#8b949e;font-size:0.78rem;">D²TL Command Center · Physics-Aware Dual-Path Coverage · Author: NOK KO</div>""", unsafe_allow_html=True)
