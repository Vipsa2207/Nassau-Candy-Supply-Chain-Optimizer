import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

# Suppress version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nassau Candy Executive Portal", layout="wide", page_icon="🏢")

# --- EXECUTIVE CSS FOR PERFECT ALIGNMENT & THEME ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    
    /* Force Column Alignment to Top */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: flex-start !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 16px 20px;
        border-radius: 10px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Uniform Title Styling - Ensures horizontal alignment */
    .section-header {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 700;
        border-left: 5px solid #3b82f6;
        padding-left: 15px;
        margin-top: 15px;
        margin-bottom: 25px;
        height: 30px;
        display: flex;
        align-items: center;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #1e293b;
        border-radius: 4px;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155 !important;
        color: #f8fafc !important;
    }

    /* Sidebar Spacing */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #334155;
    }

    /* Sidebar dropdowns: visible box on the dark sidebar */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('candy_model.pkl')
    cols = joblib.load('model_columns.pkl')
    data = pd.read_csv('cleaned_candy_data.csv')
    return model, cols, data

model, model_columns, df = load_assets()

# Core Data
factory_coords = {
    "Lot's O' Nuts": {'lat': 32.881893, 'lon': -111.768036},
    "Wicked Choccy's": {'lat': 32.076176, 'lon': -81.088371},
    "Sugar Shack": {'lat': 48.11914, 'lon': -96.18115},
    "Secret Factory": {'lat': 41.446333, 'lon': -90.565487},
    "The Other Factory": {'lat': 35.1175, 'lon': -89.971107}
}

state_coords = {
    'Texas': [31.054487, -97.563461], 'New York': [42.165726, -74.948051],
    'California': [36.116203, -119.681564], 'Illinois': [40.349457, -88.986137],
    'Florida': [27.766279, -81.686783], 'Washington': [47.400902, -121.490494],
    'Pennsylvania': [40.590752, -77.209755], 'Georgia': [33.040619, -83.643074]
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# --- SIDEBAR: CLEAN CONTROLS ---
with st.sidebar:
    st.image('logo.jpg', width=200)
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Optimization Console")
    st.markdown("---")
    
    selected_product = st.selectbox(
        "📦 Product selector", sorted(df['Product Name'].unique()),
        help="The product changes the estimated profit only. It does not change the predicted lead time or which factory is recommended.")
    selected_region = st.selectbox("📍 Region selector", sorted(list(state_coords.keys())))
    ship_mode = st.selectbox("🚚 Ship mode filter", ["Standard Class", "First Class", "Second Class", "Same Day"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**⚖️ Optimization priority slider**")
    # GRANULAR CONTINUOUS SLIDER
    opt_weight = st.slider("Priority Balance", 0, 100, 50, label_visibility="collapsed")
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; color: #94a3b8; font-size: 0.8rem; margin-top: -10px;">
            <span>Speed Focused</span>
            <span>Profit Focused</span>
        </div>
        <p style="text-align: center; color: #3b82f6; font-size: 0.85rem; font-weight: bold; margin-top: 5px;">
            Current Balance: {100-opt_weight}% Speed | {opt_weight}% Profit
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

# --- SIMULATION & NORMALIZATION LOGIC ---
results = []
for f, coords in factory_coords.items():
    dist = haversine(coords['lat'], coords['lon'], state_coords[selected_region][0], state_coords[selected_region][1])
    input_row = pd.DataFrame(0, index=[0], columns=model_columns)
    input_row['Distance_km'] = dist
    input_row['Sales'] = df['Sales'].mean()
    if f'Ship Mode_{ship_mode}' in model_columns:
        input_row[f'Ship Mode_{ship_mode}'] = 1
    
    pred_time = model.predict(input_row)[0]
    # Profit logic based on distance penalty
    est_profit = df[df['Product Name'] == selected_product]['Gross Profit'].mean() - (dist * 0.05)
    results.append({'Factory': f, 'Lead Time': pred_time, 'Distance': dist, 'Profit': est_profit})

res_df = pd.DataFrame(results)

# NORMALIZATION FOR CONTINUOUS SLIDER MATH
# Min-Max scaling to make Time and Profit comparable
res_df['t_norm'] = (res_df['Lead Time'] - res_df['Lead Time'].min()) / (res_df['Lead Time'].max() - res_df['Lead Time'].min() + 0.001)
res_df['p_norm'] = (res_df['Profit'] - res_df['Profit'].min()) / (res_df['Profit'].max() - res_df['Profit'].min() + 0.001)

# Combined Strategic Score (Lower is better)
# If opt_weight is 0 -> focus is 100% on t_norm (Time)
# If opt_weight is 100 -> focus is 100% on p_norm (Profit)
p_weight = opt_weight / 100.0
s_weight = 1.0 - p_weight
res_df['Score'] = (res_df['t_norm'] * s_weight) - (res_df['p_norm'] * p_weight)

res_df = res_df.sort_values('Score')
best_f = res_df.iloc[0]

# --- ALIGNED HEADER ---
h_col1, h_col2 = st.columns([1, 5], vertical_alignment="center")
with h_col1:
    st.image('logo.jpg', width=160)
with h_col2:
    st.markdown("""
        <div style="line-height: 1.2;">
            <h2 style="margin: 0; color: #f8fafc; font-weight: 800;">Factory Reallocation & Shipping Optimization Recommendation System</h2>
            <p style="margin: 0; color: #94a3b8; font-size: 1.1rem;">Analysis Framework for Nassau Candy Distributor</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- KPI STRIP ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Recommended Factory", best_f['Factory'])
k2.metric("Predicted Lead Time", f"{best_f['Lead Time']:.1f} days")
k3.metric("Distance to Region", f"{best_f['Distance']:,.0f} km")
profit = best_f['Profit']
k4.metric("Estimated Profit*", f"{'−' if profit < 0 else ''}${abs(profit):,.2f}")
st.caption("*Estimated profit = the product's average gross profit minus an assumed shipping cost of $0.05 per km. The rate is an assumption, not a value from the data.")

st.markdown("<br>", unsafe_allow_html=True)

# --- MODULE TABS ---
t1, t2, t3, t4 = st.tabs(["📊 Dashboard Modules", "🎯 Scenario Analysis", "⚠️ Risk Panel", "📈 Performance Metrics"])

with t1:
    # ROW 1: Level Titles
    t_col1, t_col2 = st.columns([1, 1.2])
    t_col1.markdown('<div class="section-header">🏢 Factory Optimization Simulator</div>', unsafe_allow_html=True)
    t_col2.markdown('<div class="section-header">📜 Recommendation Dashboard</div>', unsafe_allow_html=True)
    
    # ROW 2: Level Content
    c_col1, c_col2 = st.columns([1, 1.2], gap="large")
    with c_col1:
        table = res_df[['Factory', 'Lead Time', 'Distance', 'Profit']].rename(
            columns={'Lead Time': 'Lead Time (days)', 'Distance': 'Distance (km)', 'Profit': 'Est. Profit ($)'})
        st.dataframe(
            table.style.background_gradient(cmap='Blues_r', subset=['Lead Time (days)'])
                 .format({'Lead Time (days)': '{:.1f}', 'Distance (km)': '{:,.0f}', 'Est. Profit ($)': '{:,.2f}'}),
            width="stretch", hide_index=True)
    with c_col2:
        st.success(f"Strategy: Optimal Performance via {best_f['Factory']}")
        fig_reco = px.bar(res_df, x='Factory', y='Lead Time', color='Profit', template='plotly_dark', color_continuous_scale='Blues')
        fig_reco.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_reco, width="stretch")

with t2:
    # ROW 1: Level Titles
    t_col3, t_col4 = st.columns([1.5, 1])
    t_col3.markdown('<div class="section-header">🔍 What-If Scenario Analysis</div>', unsafe_allow_html=True)
    t_col4.markdown('<div class="section-header">📉 Distance Correlation Analysis</div>', unsafe_allow_html=True)
    
    # ROW 2: Level Content
    c_col3, c_col4 = st.columns([1.5, 1], gap="large")
    with c_col3:
        # FIX: Plotly 7 removed px.scatter_mapbox -> use px.scatter_map (and map_style)
        fig_map = px.scatter_map(
            res_df,
            lat=[factory_coords[f]['lat'] for f in res_df['Factory']],
            lon=[factory_coords[f]['lon'] for f in res_df['Factory']],
            color="Lead Time",
            size="Distance",
            zoom=3,
            height=480,
            map_style="carto-darkmatter",
        )
        st.plotly_chart(fig_map, width="stretch")
    with c_col4:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 7.2))
        sns.regplot(data=res_df, x="Distance", y="Lead Time", color="#3b82f6", ax=ax)
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        st.pyplot(fig)

with t3:
    st.markdown('<div class="section-header">🛡️ Risk & Impact Panel</div>', unsafe_allow_html=True)
    
    if best_f['Profit'] < 0:
        st.error("🚨 **Profit impact alert:** Margin erosion detected for this product-factory configuration.")
    elif best_f['Distance'] > 2200:
        st.warning("⚠️ **Logistical Threshold Alert:** Reallocation distance exceeds standard efficiency markers.")
    else:
        st.success("✅ **Configuration Verified:** Strategy provides optimal operational and financial stability.")
    
    st.caption("Alerts use the assumed $0.05 per km shipping cost. At that rate most long routes show a negative estimated profit, so treat these alerts as a warning about cost, not a measured loss.")
    st.markdown("<br>", unsafe_allow_html=True)
    fig_risk, ax_risk = plt.subplots(figsize=(16, 6))
    sns.barplot(data=res_df.sort_values('Profit'), x="Factory", y="Profit", palette="crest", ax=ax_risk)
    ax_risk.set_facecolor('#0f172a')
    fig_risk.patch.set_facecolor('#0f172a')
    plt.title("Impact Analysis: Profitability by Factory Route", color='#94a3b8', fontsize=14)
    st.pyplot(fig_risk)

with t4:
    # ROW 1: Level Titles
    t_col5, t_col6 = st.columns(2)
    t_col5.markdown('<div class="section-header">📈 Performance Distribution</div>', unsafe_allow_html=True)
    t_col6.markdown('<div class="section-header">📊 Model Reliability Metrics</div>', unsafe_allow_html=True)
    
    # ROW 2: Level Content
    c_col5, c_col6 = st.columns(2, gap="large")
    with c_col5:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        sns.histplot(df['Lead Time'], kde=True, color="#3b82f6", ax=ax_dist)
        st.pyplot(fig_dist)
    with c_col6:
        st.info("**Random Forest on a 20% test set:** MAE 0.87 days | RMSE 1.11 days | R² 0.59")
        st.caption("A simple average lead time per ship mode scores MAE 0.82 days and R² 0.66, so ship mode explains most of the variation. Full comparison in train_model.ipynb.")
        fig_box, ax_box = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x="Ship Mode", y="Lead Time", palette="Blues", ax=ax_box)
        st.pyplot(fig_box)
