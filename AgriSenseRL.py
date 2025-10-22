# AgriSense-AEI: Self-Learning Econometric Grassland App (PastureBase-class)
# -----------------------------------------------------------------------------
# Single-file Streamlit app with:
# - PBI-class features (grass wedge, rotation planners, grazing planner, budgets,
#   fertiliser/slurry/lime logs, soil tests, reseed/clover, reports, benchmarking)
# - Decision engine: OLS, dynamic panel (FE as System-GMM scaffold), SEM (optional),
#   Markov + Monte Carlo projections, and tabular Q-learning RL
# - NLP explanations on each chart/model and a sidebar Decision Agent (GPT‑4o‑mini)
# - Leaflet map of Ireland on the FIRST PAGE; controls live under the Planner Controls
# - "Extending to production" blueprint
#
# Usage
#   pip install streamlit plotly pandas numpy scipy scikit-learn statsmodels linearmodels semopy folium streamlit-folium
#   streamlit run Agrisense.py
#   (Optional) Add OPENAI_API_KEY to .streamlit/secrets.toml or env to enable AI text
# -----------------------------------------------------------------------------

import os
import io
import math
import json
import base64
from typing import Tuple

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# Optional OpenAI for NLP explanations
OPENAI_AVAILABLE = False
MODEL_NAME = "gpt-4o-mini"
try:
    import openai  # type: ignore
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        openai.api_key = api_key
        OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional Folium (Leaflet) for maps
FOLIUM_AVAILABLE = False
try:
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# -----------------------------------------------------------------------------
# simulated Demo Data Generators
# -----------------------------------------------------------------------------

np.random.seed(42)

PADDOCKS = [f"P{str(i).zfill(2)}" for i in range(1, 21)]
SOIL_TYPES = ["Free-draining","Moderate","Heavy"]
WEEKS = pd.date_range("2025-02-01","2025-11-30", freq="W-SUN")

# Irish county centroids (approx)
IE_COUNTY_COORDS = {
    "Carlow": (52.8365, -6.9341),
    "Cavan": (53.9908, -7.3606),
    "Clare": (52.8436, -9.0036),
    "Cork": (51.8986, -8.4756),
    "Donegal": (54.9000, -8.0000),
    "Dublin": (53.3498, -6.2603),
    "Galway": (53.2707, -9.0568),
    "Kerry": (52.1545, -9.5669),
    "Kildare": (53.1589, -6.9095),
    "Kilkenny": (52.6541, -7.2448),
    "Laois": (53.0340, -7.2990),
    "Leitrim": (53.9890, -8.0667),
    "Limerick": (52.6638, -8.6267),
    "Longford": (53.7270, -7.7970),
    "Louth": (53.9500, -6.5400),
    "Mayo": (53.8500, -9.2833),
    "Meath": (53.6520, -6.6810),
    "Monaghan": (54.2490, -6.9680),
    "Offaly": (53.2740, -7.4900),
    "Roscommon": (53.6270, -8.1890),
    "Sligo": (54.2700, -8.4700),
    "Tipperary": (52.4739, -7.9407),
    "Waterford": (52.2593, -7.1101),
    "Westmeath": (53.5250, -7.3380),
    "Wexford": (52.3369, -6.4633),
    "Wicklow": (52.9800, -6.0400),
    "Tyrone (NI)": (54.6000, -7.0000),
    "Derry (NI)": (55.0000, -7.3000),
}
COUNTIES = sorted(list(IE_COUNTY_COORDS.keys()))

# Random mapping paddock -> county (so county filters have data)
PADDOCK_TO_COUNTY = {p: np.random.choice(COUNTIES) for p in PADDOCKS}


def synth_farm_state() -> pd.DataFrame:
    rows = []
    for p in PADDOCKS:
        area = np.random.uniform(1.5, 4.0)
        soil = np.random.choice(SOIL_TYPES, p=[0.5,0.3,0.2])
        prg = np.random.uniform(0.6, 1.0)
        county = PADDOCK_TO_COUNTY[p]
        for w in WEEKS:
            temp = 7 + 10*np.sin(2*np.pi*(w.dayofyear/365)) + np.random.normal(0,1.5)
            rain = max(0, np.random.gamma(2, 5))
            fertN = np.random.choice([0,10,20,30,40,60], p=[0.25,0.15,0.2,0.2,0.15,0.05])
            soil_mod = {"Free-draining":1.0, "Moderate":0.9, "Heavy":0.8}[soil]
            growth = max(0, np.random.normal(45*soil_mod + 0.9*temp + 0.15*rain + 0.12*fertN + 15*prg, 8))
            cover = max(300, np.random.normal(1800, 350))
            demand = np.random.normal(45, 6)
            rows.append([p, w, area, soil, prg, temp, rain, fertN, growth, cover, demand, county])
    return pd.DataFrame(rows, columns=[
        "paddock","week","area_ha","soil","prg_score","temp_c","rain_mm","fertN_kg","growth_kgdm_ha","cover_kgdm_ha","demand_kgdm_ha","county"
    ])


def synth_events() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fert_rows, graze_rows, reseed_rows = [], [], []
    for p in PADDOCKS:
        for w in np.random.choice(WEEKS, size=10, replace=False):
            typ = np.random.choice(["Urea","CAN","Slurry","Lime"], p=[0.4,0.3,0.25,0.05])
            kgN = 0 if typ in ["Lime","Slurry"] else np.random.choice([20,30,40])
            fert_rows.append([p, w, typ, kgN])
        for w in np.random.choice(WEEKS, size=8, replace=False):
            pre = float(np.clip(np.random.normal(1600, 250), 1100, 2600))
            post = float(np.clip(np.random.normal(180, 80), 50, 600))
            graze_rows.append([p, w, pre, post])
        if np.random.rand() < 0.15:
            reseed_rows.append([p, np.random.choice(WEEKS), np.random.choice(["Diploid PRG","Tetraploid PRG","Clover Mix","Multi-species"])])
    fert = pd.DataFrame(fert_rows, columns=["paddock","date","type","kgN"])
    graze = pd.DataFrame(graze_rows, columns=["paddock","date","precover","postcover"])
    reseed = pd.DataFrame(reseed_rows, columns=["paddock","date","mix"])
    return fert, graze, reseed


def synth_herd() -> pd.DataFrame:
    rows = []
    for w in WEEKS:
        cows = 120 + int(np.random.normal(0,3))
        ms = float(np.random.normal(1.85, 0.15))
        intake = float(np.random.normal(16.5, 1.2))
        rows.append([w, cows, ms, intake])
    return pd.DataFrame(rows, columns=["week","cows","milk_solids_kgpd","intake_dm_kgpd"])


def synth_soil_reseed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    soil = pd.DataFrame({
        "paddock": PADDOCKS,
        "pH": np.round(np.random.normal(6.1, 0.3, len(PADDOCKS)), 2),
        "P_index": np.random.choice([1,2,3,4], len(PADDOCKS), p=[0.15,0.35,0.35,0.15]),
        "K_index": np.random.choice([1,2,3,4], len(PADDOCKS), p=[0.2,0.3,0.35,0.15]),
        "Mg_index": np.random.choice([1,2,3,4], len(PADDOCKS), p=[0.25,0.4,0.25,0.1])
    })
    reseed = pd.DataFrame({
        "paddock": np.random.choice(PADDOCKS, 6, replace=False),
        "mix": ["Diploid PRG","Tetraploid PRG","Clover mix","Clover oversow","Hybrid ryegrass","Multi-species"],
        "target_date": pd.date_range("2025-04-01", periods=6, freq="MS")
    })
    return soil, reseed


def synth_silage_budget() -> pd.DataFrame:
    return pd.DataFrame({
        "cut": pd.to_datetime(["2025-05-25","2025-07-10","2025-08-28"]),
        "bales": [220, 180, 140],
        "tDM": [45.0, 36.0, 28.0]
    })

# Create data
DF_PANEL = synth_farm_state()
DF_FERT, DF_GRAZE, DF_RESEED = synth_events()
DF_HERD = synth_herd()
DF_SOIL, DF_RESEED_PLAN = synth_soil_reseed()
DF_SILAGE = synth_silage_budget()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'


def ai_explain(prompt: str) -> str:
    if not OPENAI_AVAILABLE:
        return "AI explanation unavailable (set OPENAI_API_KEY)."
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a concise Teagasc-style agri-economist. Explain charts and model outputs clearly for farm decisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI explanation unavailable: {e}"


def grass_wedge(df_week: pd.DataFrame) -> go.Figure:
    df = df_week.copy()
    df["target"] = df["demand_kgdm_ha"].mean() * 100
    df = df.groupby("paddock", as_index=False).agg({"cover_kgdm_ha":"mean","area_ha":"first","target":"mean"})
    df["cover_total"] = df["cover_kgdm_ha"] * df["area_ha"]
    df = df.sort_values("cover_total")
    fig = go.Figure()
    fig.add_bar(x=df["paddock"], y=df["cover_total"], name="Cover (kg DM)")
    fig.add_scatter(x=df["paddock"], y=df["target"], name="Target", mode="lines")
    fig.update_layout(height=420, title="Grass Wedge (Total Cover vs Target)")
    return fig


def projected_wedge(df: pd.DataFrame, weeks_ahead=2, sims=500) -> go.Figure:
    df_last = df[df["week"]==df["week"].max()].copy()
    trans = np.array([[0.65,0.3,0.05],[0.2,0.6,0.2],[0.1,0.35,0.55]])
    state_idx = 1
    growth_mu = max(1e-6, float(df_last["growth_kgdm_ha"].mean()))
    growth_sd = float(df_last["growth_kgdm_ha"].std() or 1.0)
    totals = []
    for _ in range(sims):
        idx = state_idx
        total = 0.0
        for _h in range(weeks_ahead):
            idx = np.random.choice([0,1,2], p=trans[idx])
            mult = [0.7,1.0,1.3][idx]
            g = max(0.0, np.random.normal(growth_mu*mult, growth_sd))
            total += g
        totals.append(total)
    q10, q50, q90 = np.quantile(totals, [0.1,0.5,0.9])
    fig = go.Figure()
    fig.add_box(y=totals, name=f"{weeks_ahead}-week proj growth")
    fig.add_hline(y=q10, line_dash="dot", annotation_text="P10")
    fig.add_hline(y=q50, line_dash="dash", annotation_text="Median")
    fig.add_hline(y=q90, line_dash="dot", annotation_text="P90")
    fig.update_layout(height=380, title="Projected Wedge: Monte Carlo Growth Bands")
    return fig

# -----------------------------------------------------------------------------
# Econometric Blocks (OLS, FE as System-GMM scaffold), SEM optional
# -----------------------------------------------------------------------------

def ols_growth(df: pd.DataFrame):
    X = df[["temp_c","rain_mm","fertN_kg","prg_score"]]
    X = sm.add_constant(X)
    y = df["growth_kgdm_ha"]
    return sm.OLS(y, X).fit()


def system_gmm_panel(df: pd.DataFrame):
    # FE panel as scaffold for true AB/BB System-GMM
    d = df[["paddock","week","growth_kgdm_ha","fertN_kg","rain_mm","temp_c"]].copy()
    d = d.set_index(["paddock","week"]).sort_index()
    d["growth_lag1"] = d.groupby(level=0)["growth_kgdm_ha"].shift(1)
    d = d.dropna()
    Y = d["growth_kgdm_ha"]
    X = sm.add_constant(d[["growth_lag1","fertN_kg","rain_mm","temp_c"]])
    try:
        res = PanelOLS(Y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
        return res
    except Exception as e:
        class _Dummy:
            summary = f"Panel fit failed: {e}"
        return _Dummy()

# SEM optional
try:
    from semopy import Model as SEMModel
except Exception:
    SEMModel = None

def sem_latent_soil(df: pd.DataFrame):
    if SEMModel is None:
        return None, "SEMModel unavailable (pip install semopy)"
    model_desc = (
        "SoilFert =~ prg_score + rain_mm + temp_c\n"
        "growth_kgdm_ha ~ SoilFert + fertN_kg"
    )
    m = SEMModel(model_desc)
    try:
        m.fit(df[["prg_score","rain_mm","temp_c","growth_kgdm_ha","fertN_kg"]].dropna())
        return m, None
    except Exception as e:
        return None, str(e)

# -----------------------------------------------------------------------------
# RL: Tabular Q-learning for rotation length & N strategy
# -----------------------------------------------------------------------------

ACTIONS = [
    {"rotation_days": 16, "N_plan": "Low"},
    {"rotation_days": 18, "N_plan": "Medium"},
    {"rotation_days": 21, "N_plan": "High"},
]
STATE_BINS = {"avg_growth": [20, 40, 60, 80], "avg_cover": [1000, 1500, 2000, 2500]}
Q = {}


def _state_key(growth: float, cover: float, soil_idx: int = 1):
    g_bin = int(np.digitize(growth, STATE_BINS["avg_growth"]))
    c_bin = int(np.digitize(cover, STATE_BINS["avg_cover"]))
    return (g_bin, c_bin, soil_idx)


def _reward(growth: float, demand: float, fertN: float, env_weight: float = 0.2) -> float:
    return -abs(growth - demand) - env_weight * fertN


def q_learn_epoch(df: pd.DataFrame, alpha=0.2, gamma=0.9, eps=0.1):
    global Q
    agg = df.groupby("week", as_index=False).agg({
        "growth_kgdm_ha":"mean","cover_kgdm_ha":"mean","demand_kgdm_ha":"mean"
    })
    for _, r in agg.iterrows():
        s = _state_key(float(r.growth_kgdm_ha), float(r.cover_kgdm_ha))
        if s not in Q:
            Q[s] = np.zeros(len(ACTIONS))
        a_idx = np.random.randint(len(ACTIONS)) if np.random.rand() < eps else int(np.argmax(Q[s]))
        act = ACTIONS[a_idx]
        fertN = {"Low":10, "Medium":25, "High":40}[act["N_plan"]]
        demand_adj = float(r.demand_kgdm_ha) * (18.0 / act["rotation_days"])
        rwd = _reward(float(r.growth_kgdm_ha), demand_adj, fertN)
        s2 = s
        Q[s][a_idx] = Q[s][a_idx] + alpha * (rwd + gamma * np.max(Q[s2]) - Q[s][a_idx])


def rl_recommendation(df: pd.DataFrame):
    g = float(df["growth_kgdm_ha"].tail(30).mean())
    c = float(df["cover_kgdm_ha"].tail(30).mean())
    s = _state_key(g, c)
    if s not in Q or np.allclose(Q.get(s, np.zeros(len(ACTIONS))), 0):
        for _ in range(50):
            q_learn_epoch(df)
    best = int(np.argmax(Q.get(s, np.zeros(len(ACTIONS)))))
    return ACTIONS[best], s, Q.get(s, np.zeros(len(ACTIONS)))

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="AgriSense-AEI (PastureBase-class)", layout="wide")
st.title("AgriSense— Self-Learning Econometric Decision System for AgriFood Business Ireland")
st.caption("Prototype System designed by Shubhojit Bagchi")

# Sidebar controls
st.sidebar.header("Planner Controls")
sel_week = st.sidebar.select_slider("Week", options=sorted(DF_PANEL["week"].unique()))
weeks_ahead = st.sidebar.slider("Projection horizon (weeks)", 1, 6, 2)
N_plan = st.sidebar.selectbox("Nitrogen plan", ["Low","Medium","High"], index=1)
rotation_days = st.sidebar.slider("Rotation length (days)", 14, 24, 18)

st.sidebar.subheader("Map Controls")
county = st.sidebar.selectbox("County", options=COUNTIES, index=COUNTIES.index("Dublin") if "Dublin" in COUNTIES else 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Decision Agent (GPT‑4o‑mini)")
agent_enabled = st.sidebar.checkbox("Enable AI agent", value=False)
agent_q = st.sidebar.text_area("Ask a question (e.g., ‘How to adjust rotation if growth drops 15%?’)")
if agent_enabled and agent_q:
    dfw_ctx = DF_PANEL[DF_PANEL["week"]==sel_week]
    kpi = {
        "avg_growth": float(dfw_ctx["growth_kgdm_ha"].mean()),
        "avg_cover": float(dfw_ctx["cover_kgdm_ha"].mean()),
        "mean_N": float(dfw_ctx["fertN_kg"].mean()),
        "demand": float(dfw_ctx["demand_kgdm_ha"].mean()),
        "rotation_days": int(rotation_days),
        "N_plan": N_plan,
    }
    prompt = (
        f"Given KPIs {kpi}, advise a practical weekly plan "
        f"(rotation, N, grazing order) with risks and contingencies.\n"
        f"Question: {agent_q}"
    )
    st.sidebar.info(ai_explain(prompt))

# --------------------------- FIRST PAGE: MAP ---------------------------------
st.subheader("Ireland Map — Selected County Marker")
if not FOLIUM_AVAILABLE:
    st.warning("Install map deps: pip install folium streamlit-folium")
else:
    lat, lon = IE_COUNTY_COORDS.get(county, (53.4, -8.2))
    # Compute county-week estimates
    county_week = DF_PANEL[(DF_PANEL["county"]==county) & (DF_PANEL["week"]==sel_week)]
    if county_week.empty:
        county_week = DF_PANEL[DF_PANEL["week"]==sel_week]
    est_growth = float(county_week["growth_kgdm_ha"].mean())
    est_cover = float(county_week["cover_kgdm_ha"].mean())
    popup = f"{county}: Est. growth {est_growth:.1f} kgDM/ha/d · cover {est_cover:.0f} kgDM/ha"
    m = folium.Map(location=[lat, lon], zoom_start=7)
    folium.Marker([lat, lon], popup=popup, tooltip=county).add_to(m)
    folium.CircleMarker([lat, lon], radius=10, fill=True, fill_opacity=0.2).add_to(m)
    st_folium(m, width=None, height=420)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Estimated Growth (kgDM/ha/d)", f"{est_growth:.1f}")
    with c2:
        st.metric("Estimated Cover (kgDM/ha)", f"{est_cover:.0f}")
    st.caption("County selector lives in the Planner Controls. Marker popup shows the county and estimated KPIs for the selected week.")

with st.expander("ℹ️ Data & Import/Export", expanded=True):
    st.write("This demo uses simulated data. Upload CSVs to replace panels/events/herd.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        up_panel = st.file_uploader("Panel (paddock×week)", type=["csv"])
        if up_panel:
            DF_PANEL = pd.read_csv(up_panel, parse_dates=["week"])  # noqa
    with c2:
        up_events = st.file_uploader("Fertiliser/Slurry events", type=["csv"])
        if up_events:
            DF_FERT = pd.read_csv(up_events, parse_dates=["date"])  # noqa
    with c3:
        up_graze = st.file_uploader("Grazing events", type=["csv"])
        if up_graze:
            DF_GRAZE = pd.read_csv(up_graze, parse_dates=["date"])  # noqa
    with c4:
        up_soil = st.file_uploader("Soil tests (optional)", type=["csv"])
        if up_soil:
            DF_SOIL = pd.read_csv(up_soil)  # noqa
    st.markdown(download_link(DF_PANEL.head(200), "panel_sample.csv", "⬇️ Download sample Panel CSV"), unsafe_allow_html=True)

# Tabs
Tabs = st.tabs([
    "Grass Wedge", "Projected Wedge", "Grazing Planner (SRP/ARP)", "Demand vs Growth",
    "Milk & Budgets", "Fertiliser & Records", "Soil Tests & Reseed",
    "Analytics (OLS/GMM/SEM)", "RL Policy", "Reports & Benchmark", "Tutorials",
    "Extending to production",
])

# 1) Grass Wedge
with Tabs[0]:
    st.subheader("Grass Wedge (Current Week)")
    dfw = DF_PANEL[DF_PANEL["week"]==sel_week]
    fig = grass_wedge(dfw)
    st.plotly_chart(fig, use_container_width=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Growth (kgDM/ha/d)", f"{dfw['growth_kgdm_ha'].mean():.1f}")
    k2.metric("Avg Cover (kgDM/ha)", f"{dfw['cover_kgdm_ha'].mean():.0f}")
    k3.metric("Mean N this week (kg/ha)", f"{dfw['fertN_kg'].mean():.1f}")
    k4.metric("Demand (kgDM/ha/d)", f"{dfw['demand_kgdm_ha'].mean():.1f}")
    st.markdown("**AI explanation:** " + ai_explain("Explain how a grass wedge guides paddock order and optimal residuals for utilisation and performance."))

# 2) Projected Wedge
with Tabs[1]:
    st.subheader("Projected Wedge (Monte Carlo + Markov)")
    figp = projected_wedge(DF_PANEL, weeks_ahead=weeks_ahead, sims=500)
    st.plotly_chart(figp, use_container_width=True)
    st.info("Projection bands reflect uncertainty from state transitions (Low/Normal/High growth) and yield variance.")
    st.markdown("**AI explanation:** " + ai_explain("Explain P10/Median/P90 bands and actions under low vs high growth (rotation, supplementation)."))

# 3) Grazing Planner (SRP/ARP)
with Tabs[2]:
    st.subheader("Spring & Autumn Rotation Planners (SRP/ARP)")
    d = DF_PANEL[DF_PANEL["week"]==sel_week].copy()
    d["cover_total"] = d["cover_kgdm_ha"] * d["area_ha"]
    order = d.sort_values("cover_total", ascending=False)[["paddock","area_ha","cover_total"]]
    order["cum_area_%"] = order["area_ha"].cumsum()/order["area_ha"].sum()*100
    figg = px.line(order, x=order.index, y="cum_area_%", markers=True, title="Cumulative Area (grazed/closed) vs paddock order")
    st.plotly_chart(figg, use_container_width=True)
    st.markdown("**AI guidance:** " + ai_explain("Explain SRP and ARP (60:40 closing) and how rotation length affects next spring supply."))

# 4) Demand vs Growth
with Tabs[3]:
    st.subheader("Demand vs Growth & Stocking Rate")
    merged = d.merge(DF_HERD, on="week", how="left")
    merged["SR_cows_ha"] = merged["cows"].sum()/merged["area_ha"].sum()
    merged["demand_kgdm_ha_d"] = 45*(18/rotation_days)
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Growth kgDM/ha/d", f"{merged['growth_kgdm_ha'].mean():.1f}")
    k2.metric("Demand kgDM/ha/d", f"{merged['demand_kgdm_ha_d'].mean():.1f}")
    k3.metric("Stocking Rate cows/ha", f"{merged['SR_cows_ha'].mean():.2f}")
    st.markdown("**AI explanation:** " + ai_explain("Explain balancing demand with growth, the role of SR, and when to add/remove supplementation."))

# 5) Milk & Budgets
with Tabs[4]:
    st.subheader("Milk & Winter Fodder Budget")
    st.dataframe(DF_HERD)
    st.dataframe(DF_SILAGE)
    tdm_req = float(DF_HERD["cows"].mean()*13.5*120/1000)  # 120 days @ 13.5 kgDM/d per cow
    tdm_have = float(DF_SILAGE["tDM"].sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("t DM required (est)", f"{tdm_req:.1f}")
    k2.metric("t DM available", f"{tdm_have:.1f}")
    k3.metric("Balance (t)", f"{(tdm_have-tdm_req):.1f}")
    st.markdown("**AI advice:** " + ai_explain("Assess fodder balance given cows, winter length, and silage; suggest actions if deficit (extra bales, wholecrop, reduce SR)."))

# 6) Fertiliser & Records
with Tabs[5]:
    st.subheader("Nutrient Applications & Events")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Fertiliser & Slurry**")
        st.dataframe(DF_FERT.sort_values("date"))
    with c2:
        st.write("**Grazing Events**")
        st.dataframe(DF_GRAZE.sort_values("date"))
    st.write("**Reseed Records**")
    st.dataframe(DF_RESEED.sort_values("date"))
    st.markdown("**AI explanation:** " + ai_explain("Discuss N strategy (Low/Med/High), slurry/lime timing, NUE with environmental safeguards."))

# 7) Soil Tests & Reseed
with Tabs[6]:
    st.subheader("Soil Tests (pH, P, K, Mg) & Reseed/Clover Planner")
    st.dataframe(DF_SOIL)
    st.dataframe(DF_RESEED_PLAN)
    st.markdown("**AI recommendation:** " + ai_explain("Interpret soil test indices for lime/P/K management and when to oversow clover vs full reseed."))

# 8) Analytics
with Tabs[7]:
    st.subheader("Econometric Analytics (OLS · FE scaffold for System-GMM · SEM)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**OLS: Growth ~ Temp + Rain + FertN + PRG**")
        ols_res = ols_growth(DF_PANEL)
        st.text(ols_res.summary().as_text())
        st.markdown("**AI interpretation:** " + ai_explain("Interpret OLS coefficients and significance for grazing/fertiliser decisions."))
    with c2:
        st.markdown("**Dynamic Panel (FE as placeholder for System GMM)**")
        gmm_res = system_gmm_panel(DF_PANEL)
        st.text(str(gmm_res.summary))
        st.markdown("**AI interpretation:** " + ai_explain("Explain lagged growth and controls; note AB/BB diagnostics to add for true System-GMM."))
    with st.expander("SEM (latent Soil Fertility) — optional"):
        m, err = sem_latent_soil(DF_PANEL)
        if err:
            st.warning(f"SEM not run: {err}")
        else:
            st.write("Path estimates:")
            st.write(m.inspect())
            st.markdown("**AI explanation:** " + ai_explain("Explain latent SoilFert links and identification cautions; data needs (soil tests)."))

# 9) RL Policy
with Tabs[8]:
    st.subheader("RL Policy: Rotation & N Strategy")
    if st.button("Train RL on last season"):
        for _ in range(50):
            q_learn_epoch(DF_PANEL)
        st.success("RL training complete (tabular Q-learning)")
    rec, s_key, qvals = rl_recommendation(DF_PANEL)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Rotation (days)", rec["rotation_days"])
    c2.metric("N Plan", rec["N_plan"])
    c3.metric("State key", str(s_key))
    st.bar_chart(pd.DataFrame(qvals, index=["LowN_16","MedN_18","HighN_21"]))
    st.markdown("**AI decision rationale:** " + ai_explain(
        f"Given recent growth and covers, justify rotation {rec['rotation_days']} days and N plan {rec['N_plan']} with trade-offs and risks."
    ))

# 10) Reports & Benchmark
with Tabs[9]:
    st.subheader("Annual Tonnage · Grazings/yr · Peer Benchmark")
    d_all = DF_PANEL.copy(); d_all["year"] = d_all["week"].dt.year
    tonnage = d_all.groupby(["year","paddock"], as_index=False)["growth_kgdm_ha"].sum()
    figt = px.box(tonnage, x="year", y="growth_kgdm_ha", points="all", title="Annual Tonnage by Paddock")
    st.plotly_chart(figt, use_container_width=True)
    grazings = DF_GRAZE.groupby("paddock", as_index=False)["date"].count().rename(columns={"date":"grazings_yr"})
    st.dataframe(grazings)
    peers = tonnage.copy(); peers["farm"] = np.random.choice(["PeerA","PeerB","PeerC"], size=len(peers))
    st.write("**Peer Benchmark (simulated)**")
    st.dataframe(peers.groupby(["year","farm"], as_index=False)["growth_kgdm_ha"].mean().rename(columns={"growth_kgdm_ha":"avg_tonnage"}))
    st.markdown("**AI note:** " + ai_explain("Use grazings/year and tonnage to rank paddocks; actions for underperformers (reseed, nutrients, drainage)."))

# 11) Tutorials
with Tabs[10]:
    st.subheader("Tutorials & Workflows (PBI-style)")
    st.markdown("- Projected Wedge → set rotation & supplementation under uncertainty\n- SRP/ARP → meet weekly area targets; 60:40 closing\n- Budgets → fodder balance, supplementation triggers")
    st.markdown("**AI coach:** " + ai_explain("Provide a step-by-step on using projected wedge, rotation planners, and budgets weekly."))

# 12) Extending to production
with Tabs[11]:
    st.subheader("Extending to production")
    st.markdown(
        """
- **True System‑GMM (AB/BB):** instruments, AR(1)/AR(2), Hansen/Sargan; nightly re‑estimation.
- **Forecasting:** integrate Met Éireann 7‑day forecast; add Kalman/BVAR for growth and covers.
- **Nutrient compliance & NUE:** protected urea suggestions, lime from pH; auto‑records for N‑load.
- **Milk & budgets:** connect parlour API; live fodder balance and supplementation triggers.
- **Offline-first packaging:** Wrap with Tauri/Eel for no‑signal paddock entry; background sync.
- **Groups & benchmarking:** privacy‑preserving aggregates; weekly Grass10‑style reports.
- **Agent tuning:** fine‑tune GPT‑4o‑mini with Teagasc protocols; multi‑objective RL (profit, MS, N, risk).
- **Security/ops:** OAuth, RBAC, Docker/K8s, CI/CD, audit.
        """
    )

st.success("AgriSense-AEI ready — upload your farm CSVs or run the simulated demo. Add OPENAI_API_KEY to enable NLP & Decision Agent.")
