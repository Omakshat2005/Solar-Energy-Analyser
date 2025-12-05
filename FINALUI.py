import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

# --- Configuration ---
CSV_FILE = 'surat_weather_Finalv4_3years.csv'
PANEL_AREA_M2 = 2.0
GRID_RATE = 7.0

LOAD_PROFILE = {
    "AC (1.5 Ton)": {"qty": 1, "kwh": 1.5},
    "Fans": {"qty": 5, "kwh": 0.075},
    "LEDs": {"qty": 10, "kwh": 0.01},
    "Washing Machine": {"qty": 1, "kwh": 0.5}
}

# --- Page Config ---
st.set_page_config(
    page_title="SHEOS AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Ultra Dark Theme CSS ---
# --- Ultra Dark Theme CSS (Fixed Visibility) ---
st.markdown("""
<style>
    /* 1. GLOBAL TEXT & BACKGROUND */
    .main, .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* 2. CONTROLS (INPUTS) - FORCE WHITE BOX, BLACK TEXT */
    /* Target Date, Time, Number, and Text inputs */
    input[type="text"], input[type="number"], input[type="date"], input[type="time"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00ff9d !important;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Target Selectbox (Dropdowns) */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00ff9d !important;
    }
    /* Force text inside selectbox to be black */
    div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    /* Dropdown menu options */
    div[data-baseweb="popover"] div, ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    div[data-baseweb="option"], li[data-baseweb="option"] {
        color: #000000 !important;
    }

    /* 3. LABELS (Text ABOVE the inputs) */
    /* Make "Date", "Number of Panels", etc. Bright White */
    .stDateInput label, .stTimeInput label, .stNumberInput label, .stSelectbox label, .stTextInput label {
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
    
    /* 4. METRICS / WEATHER DETAILS */
    /* The Value (e.g., "30°C", "850 W/m²") */
    div[data-testid="stMetricValue"] {
        color: #00ff9d !important; /* Neon Green */
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.4);
    }
    /* The Label (e.g., "Temperature", "Irradiance") */
    div[data-testid="stMetricLabel"] {
        color: #00b8ff !important; /* Neon Blue */
        font-weight: bold;
    }
    /* The container for the metrics */
    div[data-testid="metric-container"] {
        background-color: #111111;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
    }

    /* 5. SIDEBAR Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #333;
    }
    /* Sidebar Text */
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li {
        color: #cccccc !important;
    }
    
    /* 6. BUTTONS */
    .stButton > button {
        background: linear-gradient(45deg, #00ff9d, #00b8ff);
        color: black;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)
# --- Data Loading & Model Training (Cached) ---
@st.cache_resource
def load_and_train_model(csv_path):
    """Load CSV and train ML model - cached for performance"""
    try:
        if not os.path.exists(csv_path):
            st.error(f"❌ **File Not Found**: `{csv_path}`")
            st.info(f"📂 **Current Directory**: `{os.getcwd()}`")
            st.warning("**Solution**: Place `surat_weather_Finalv4_3years.csv` in the same folder as app.py")
            st.stop()
        
        raw_df = pd.read_csv(csv_path)
        st.success(f"✅ Loaded {len(raw_df):,} records from CSV")
        
        column_mapping = {}
        if 'timestamp' in raw_df.columns:
            column_mapping['timestamp'] = 'datetime'
        if 'temp_C' in raw_df.columns:
            column_mapping['temp_C'] = 'temperature_C'
        if 'precipitation_probability_pct' in raw_df.columns:
            column_mapping['precipitation_probability_pct'] = 'cloud_percentage'
        
        if column_mapping:
            raw_df.rename(columns=column_mapping, inplace=True)
        
        if 'cloud_percentage' not in raw_df.columns:
            st.warning("⚠️ 'cloud_percentage' column not found. Using default value of 30%")
            raw_df['cloud_percentage'] = 30.0
        
        required = ['datetime', 'irradiance_W_m2', 'temperature_C', 'cloud_percentage']
        df = raw_df[required].copy()
        
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
        except:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        df = df.sort_values('datetime').reset_index(drop=True)
        
        scaler = StandardScaler()
        training_data = df.copy()
        training_data['hour'] = training_data['datetime'].dt.hour
        training_data['power_output'] = training_data.apply(
            lambda row: calculate_physics_generation(row, num_panels=1), axis=1
        )
        
        features = ['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']
        X = training_data[features]
        y = training_data['power_output']
        
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        st.success(f"✅ Model trained successfully! Accuracy: {r2*100:.2f}%")
        
        return df, model, scaler, r2
        
    except Exception as e:
        st.error(f"❌ **Unexpected Error**: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
        return None, None, None, None

def calculate_physics_generation(row, num_panels):
    """Physics-based power calculation"""
    irradiance = row['irradiance_W_m2']
    cloud = row['cloud_percentage']
    
    if 0 <= cloud <= 30:
        efficiency = 1.0
    elif 30 < cloud <= 60:
        efficiency = 0.8
    elif 60 < cloud <= 90:
        efficiency = 0.6
    else:
        efficiency = 0.2
    
    total_area = num_panels * PANEL_AREA_M2
    power_kw = (irradiance * total_area * efficiency) / 1000.0
    
    return max(0.0, power_kw)

def get_prediction(model, scaler, feature_df):
    """Scale and predict"""
    scaled = scaler.transform(feature_df)
    return model.predict(scaled)

# Color palette
COLORS = {
    'primary': '#00ff88',
    'secondary': '#00d4ff',
    'accent': '#ff6b9d',
    'warning': '#ffd93d',
    'danger': '#ff6b6b',
    'success': '#6bcf7f'
}

# --- Load Data ---
df, model, scaler, model_accuracy = load_and_train_model(CSV_FILE)

# --- Sidebar ---
st.sidebar.markdown("<h2 style='text-align: center; font-size: 2rem;'>⚙️ CONTROL</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

num_panels = st.sidebar.number_input(
    "🔢 Solar Panels",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
    help="Number of solar panels installed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-size: 1.3rem;'>📅 TIME</h3>", unsafe_allow_html=True)

min_date = df['datetime'].min().date()
max_date = df['datetime'].max().date()
selected_date = st.sidebar.date_input(
    "Date",
    value=min_date + timedelta(days=365),
    min_value=min_date,
    max_value=max_date
)

selected_hour = st.sidebar.slider("Hour", 0, 23, 12)

selected_datetime = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)
current_index = (df['datetime'] - selected_datetime).abs().idxmin()
current_row = df.iloc[current_index]

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-size: 1.3rem;'>🌤️ WEATHER</h3>", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("🌡️", f"{current_row['temperature_C']:.1f}°C", help="Temperature")
    st.metric("☀️", f"{current_row['irradiance_W_m2']:.0f}", help="Irradiance W/m²")
with col2:
    st.metric("☁️", f"{current_row['cloud_percentage']:.0f}%", help="Cloud Cover")
    st.metric("🤖", f"{model_accuracy*100:.1f}%", help="Model Accuracy")

# --- Header ---
st.markdown("<h1>☀️ SHEOS AI</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #888; font-size: 1.2rem; margin-top: -10px;'>{current_row['datetime'].strftime('%d %B %Y • %H:%M')}</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 DASHBOARD",
    "📈 FORECAST",
    "⚡ MONITOR",
    "🧠 SCHEDULER",
    "💰 ROI"
])

# --- TAB 1: Dashboard ---
with tab1:
    input_data = pd.DataFrame([{
        'irradiance_W_m2': current_row['irradiance_W_m2'],
        'temperature_C': current_row['temperature_C'],
        'cloud_percentage': current_row['cloud_percentage'],
        'hour': current_row['datetime'].hour
    }])
    
    current_gen = get_prediction(model, scaler, input_data).item() * num_panels
    
    cloud = current_row['cloud_percentage']
    if 0 <= cloud <= 30:
        efficiency = 100
    elif 30 < cloud <= 60:
        efficiency = 80
    elif 60 < cloud <= 90:
        efficiency = 60
    else:
        efficiency = 20
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⚡ POWER", f"{current_gen:.2f} kW", delta=f"{current_gen*0.15:.2f} vs avg")
    with col2:
        st.metric("📊 PANELS", f"{num_panels}", delta=None)
    with col3:
        st.metric("💵 RATE", f"₹{GRID_RATE}/kWh", delta=None)
    with col4:
        st.metric("✨ EFFICIENCY", f"{efficiency}%", delta=None)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Dashboard - 2 Column Layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=efficiency,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "SYSTEM EFFICIENCY", 'font': {'size': 24, 'color': '#888'}},
            number={'font': {'size': 70, 'color': COLORS['primary']}, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#333"},
                'bar': {'color': COLORS['primary'], 'thickness': 0.7},
                'bgcolor': "rgba(0, 0, 0, 0.3)",
                'borderwidth': 3,
                'bordercolor': COLORS['primary'],
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(255, 107, 107, 0.2)'},
                    {'range': [30, 60], 'color': 'rgba(255, 217, 61, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(0, 255, 136, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': COLORS['accent'], 'width': 5},
                    'thickness': 0.9,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#888'},
            margin=dict(t=80, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_right:
        # Real-time Power Meter (Speedometer style)
        max_power = num_panels * 0.4  # Max theoretical power per panel
        
        fig_speed = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_gen,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "LIVE GENERATION", 'font': {'size': 24, 'color': '#888'}},
            number={'font': {'size': 70, 'color': COLORS['secondary']}, 'suffix': ' kW'},
            delta={'reference': max_power * 0.6, 'increasing': {'color': COLORS['success']}},
            gauge={
                'axis': {'range': [None, max_power], 'tickwidth': 2, 'tickcolor': "#333"},
                'bar': {'color': COLORS['secondary'], 'thickness': 0.7},
                'bgcolor': "rgba(0, 0, 0, 0.3)",
                'borderwidth': 3,
                'bordercolor': COLORS['secondary'],
                'steps': [
                    {'range': [0, max_power*0.3], 'color': 'rgba(255, 107, 107, 0.15)'},
                    {'range': [max_power*0.3, max_power*0.7], 'color': 'rgba(255, 217, 61, 0.15)'},
                    {'range': [max_power*0.7, max_power], 'color': 'rgba(0, 212, 255, 0.15)'}
                ]
            }
        ))
        
        fig_speed.update_layout(
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#888'},
            margin=dict(t=80, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Environmental Impact Visualization
    st.markdown("<h3 style='text-align: center;'>🌍 ENVIRONMENTAL IMPACT (TODAY)</h3>", unsafe_allow_html=True)
    
    daily_gen = current_gen * 8  # Assume 8 hours of good generation
    co2_saved = daily_gen * 0.82  # kg CO2 per kWh
    trees_equivalent = co2_saved / 21  # 1 tree absorbs 21kg CO2/year, daily = /365
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_co2 = go.Figure(go.Indicator(
            mode="number+delta",
            value=co2_saved,
            number={'suffix': " kg", 'font': {'size': 50, 'color': COLORS['success']}},
            title={'text': "CO₂ SAVED", 'font': {'size': 18, 'color': '#888'}},
            delta={'reference': co2_saved * 0.8, 'relative': False}
        ))
        fig_co2.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_co2, use_container_width=True)
    
    with col2:
        fig_trees = go.Figure(go.Indicator(
            mode="number",
            value=trees_equivalent * 365,
            number={'suffix': " 🌲", 'font': {'size': 50, 'color': COLORS['success']}},
            title={'text': "TREES PLANTED (YEARLY)", 'font': {'size': 18, 'color': '#888'}}
        ))
        fig_trees.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_trees, use_container_width=True)
    
    with col3:
        money_saved = daily_gen * GRID_RATE
        fig_money = go.Figure(go.Indicator(
            mode="number+delta",
            value=money_saved,
            number={'prefix': "₹", 'font': {'size': 50, 'color': COLORS['warning']}},
            title={'text': "MONEY SAVED", 'font': {'size': 18, 'color': '#888'}},
            delta={'reference': money_saved * 0.8, 'relative': False}
        ))
        fig_money.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_money, use_container_width=True)

# --- TAB 2: Solar Forecast ---
with tab2:
    current_dt = current_row['datetime']
    start = current_dt.replace(hour=0, minute=0, second=0)
    end = current_dt.replace(hour=23, minute=59, second=59)
    
    mask = (df['datetime'] >= start) & (df['datetime'] <= end)
    day_data = df.loc[mask].copy()
    
    if len(day_data) > 0:
        day_data['hour'] = day_data['datetime'].dt.hour
        features = day_data[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
        day_data['pred'] = get_prediction(model, scaler, features) * num_panels
        
        # Multi-metric visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Solar Generation Forecast", "Irradiance Profile", "Temperature Trend", "Cloud Coverage"),
            specs=[[{"colspan": 2}, None], [{}, {}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Main generation chart (full width)
        fig.add_trace(go.Scatter(
            x=day_data['hour'],
            y=day_data['pred'],
            fill='tozeroy',
            mode='lines+markers',
            name='Generation',
            line=dict(color=COLORS['primary'], width=4),
            fillcolor=f'rgba(0, 255, 136, 0.2)',
            marker=dict(size=10, color=COLORS['primary'], line=dict(color=COLORS['secondary'], width=2)),
            hovertemplate='<b>%{x}:00</b><br>%{y:.3f} kW<extra></extra>'
        ), row=1, col=1)
        
        # Current hour marker
        fig.add_vline(
            x=current_dt.hour,
            line_dash="dash",
            line_color=COLORS['accent'],
            line_width=3,
            annotation_text="NOW",
            row=1, col=1
        )
        
        # Irradiance
        fig.add_trace(go.Scatter(
            x=day_data['hour'],
            y=day_data['irradiance_W_m2'],
            mode='lines+markers',
            name='Irradiance',
            line=dict(color=COLORS['warning'], width=3),
            marker=dict(size=8),
            hovertemplate='%{y:.0f} W/m²<extra></extra>'
        ), row=2, col=1)
        
        # Temperature
        fig.add_trace(go.Scatter(
            x=day_data['hour'],
            y=day_data['temperature_C'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color=COLORS['danger'], width=3),
            marker=dict(size=8),
            hovertemplate='%{y:.1f}°C<extra></extra>'
        ), row=2, col=2)
        
        # Cloud coverage (inverted area)
        fig.add_trace(go.Scatter(
            x=day_data['hour'],
            y=day_data['cloud_percentage'],
            fill='tozeroy',
            mode='lines',
            name='Clouds',
            line=dict(color=COLORS['secondary'], width=2),
            fillcolor='rgba(0, 212, 255, 0.15)',
            hovertemplate='%{y:.0f}%<extra></extra>'
        ), row=2, col=1)
        
        fig.update_xaxes(title_text="Hour", row=2, col=1, gridcolor='#1a1a1a')
        fig.update_xaxes(title_text="Hour", row=2, col=2, gridcolor='#1a1a1a')
        fig.update_yaxes(title_text="Power (kW)", row=1, col=1, gridcolor='#1a1a1a')
        fig.update_yaxes(title_text="W/m² / %", row=2, col=1, gridcolor='#1a1a1a')
        fig.update_yaxes(title_text="°C", row=2, col=2, gridcolor='#1a1a1a')
        
        fig.update_layout(
            height=800,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a0a0a',
            font=dict(color='#888'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0, 255, 136, 0.05)',
                bordercolor=COLORS['primary'],
                borderwidth=1
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics with icons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("☀️ TOTAL", f"{day_data['pred'].sum():.2f} kWh", delta=f"+{day_data['pred'].sum()*0.15:.1f} vs avg")
        with col2:
            peak_hour = int(day_data.loc[day_data['pred'].idxmax(), 'hour'])
            st.metric("🔥 PEAK", f"{day_data['pred'].max():.2f} kW", delta=f"at {peak_hour}:00")
        with col3:
            avg_gen = day_data['pred'].mean()
            st.metric("📊 AVERAGE", f"{avg_gen:.2f} kW", delta=None)
        with col4:
            revenue = day_data['pred'].sum() * GRID_RATE
            st.metric("💰 VALUE", f"₹{revenue:.2f}", delta=None)

# --- TAB 3: Load Monitor ---
with tab3:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ APPLIANCES")
        st.markdown("<br>", unsafe_allow_html=True)
        
        ac_qty = st.number_input("🌨️ AC Units", 0, 10, 1, key="ac")
        fan_qty = st.number_input("🌀 Fans", 0, 20, 5, key="fan")
        led_qty = st.number_input("💡 LEDs", 0, 50, 10, key="led")
        wm_active = st.checkbox("🧺 Washing Machine", value=False, key="wm")
        
        total_load = (
            ac_qty * LOAD_PROFILE["AC (1.5 Ton)"]["kwh"] +
            fan_qty * LOAD_PROFILE["Fans"]["kwh"] +
            led_qty * LOAD_PROFILE["LEDs"]["kwh"] +
            (LOAD_PROFILE["Washing Machine"]["kwh"] if wm_active else 0)
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("🏠 TOTAL LOAD", f"{total_load:.2f} kW")
        
        # Breakdown pie chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 LOAD BREAKDOWN")
        
        loads = []
        labels = []
        colors_pie = []
        
        if ac_qty > 0:
            loads.append(ac_qty * LOAD_PROFILE["AC (1.5 Ton)"]["kwh"])
            labels.append(f"AC ({ac_qty})")
            colors_pie.append(COLORS['danger'])
        if fan_qty > 0:
            loads.append(fan_qty * LOAD_PROFILE["Fans"]["kwh"])
            labels.append(f"Fans ({fan_qty})")
            colors_pie.append(COLORS['secondary'])
        if led_qty > 0:
            loads.append(led_qty * LOAD_PROFILE["LEDs"]["kwh"])
            labels.append(f"LEDs ({led_qty})")
            colors_pie.append(COLORS['warning'])
        if wm_active:
            loads.append(LOAD_PROFILE["Washing Machine"]["kwh"])
            labels.append("Washing Machine")
            colors_pie.append(COLORS['accent'])
        
        if loads:
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=loads,
                hole=0.5,
                marker=dict(colors=colors_pie, line=dict(color='#0a0a0a', width=3)),
                textfont=dict(size=14, color='white'),
                hovertemplate='<b>%{label}</b><br>%{value:.2f} kW<br>%{percent}<extra></extra>'
            )])
            
            fig_pie.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#888'},
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        gen = get_prediction(model, scaler, input_data).item() * num_panels
        
        st.markdown("### ⚡ POWER FLOW")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison bar chart
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            name='Solar Generation',
            x=['Power Flow'],
            y=[gen],
            marker=dict(
                color=COLORS['primary'],
                line=dict(color=COLORS['secondary'], width=3),
                pattern_shape="/"
            ),
            text=[f'{gen:.2f} kW'],
            textposition='auto',
            textfont=dict(size=20, color='white', family='Arial Black'),
            width=0.4
        ))
        
        fig_bar.add_trace(go.Bar(
            name='House Load',
            x=['Power Flow'],
            y=[total_load],
            marker=dict(
                color=COLORS['accent'],
                line=dict(color=COLORS['danger'], width=3),
                pattern_shape="\\"
            ),
            text=[f'{total_load:.2f} kW'],
            textposition='auto',
            textfont=dict(size=20, color='white', family='Arial Black'),
            width=0.4
        ))
        
        fig_bar.update_layout(
            yaxis_title="Power (kW)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a0a0a',
            height=400,
            barmode='group',
            font=dict(color='#888', size=14),
            yaxis=dict(gridcolor='#1a1a1a'),
            bargap=0.3
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Net power calculation
        net = total_load - gen
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if net > 0:
            hourly_cost = net * GRID_RATE
            daily_cost = hourly_cost * 24
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.error(f"### ⚠️ GRID IMPORT")
                st.error(f"**{net:.2f} kW** needed from grid")
            with col_b:
                st.error(f"### 💸 COST")
                st.error(f"₹**{hourly_cost:.2f}**/hr • ₹**{daily_cost:.2f}**/day")
        else:
            surplus = abs(net)
            hourly_value = surplus * GRID_RATE
            daily_value = hourly_value * 24
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.success(f"### ✅ SURPLUS")
                st.success(f"**{surplus:.2f} kW** excess power")
            with col_b:
                st.success(f"### 💚 VALUE")
                st.success(f"₹**{hourly_value:.2f}**/hr • ₹**{daily_value:.2f}**/day")
        
        # Real-time gauge for self-sufficiency
        st.markdown("<br>", unsafe_allow_html=True)
        
        if total_load > 0:
            sufficiency = min((gen / total_load) * 100, 100)
        else:
            sufficiency = 100
        
        fig_suff = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sufficiency,
            title={'text': "SELF-SUFFICIENCY", 'font': {'size': 20, 'color': '#888'}},
            number={'suffix': '%', 'font': {'size': 50, 'color': COLORS['primary']}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#333"},
                'bar': {'color': COLORS['primary'], 'thickness': 0.8},
                'bgcolor': "rgba(0, 0, 0, 0.3)",
                'borderwidth': 2,
                'bordercolor': COLORS['primary'],
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 107, 107, 0.2)'},
                    {'range': [50, 80], 'color': 'rgba(255, 217, 61, 0.2)'},
                    {'range': [80, 100], 'color': 'rgba(0, 255, 136, 0.2)'}
                ]
            }
        ))
        
        fig_suff.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#888'},
            margin=dict(t=60, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_suff, use_container_width=True)

# --- TAB 4: Smart Scheduler ---
with tab4:
    current_dt = current_row['datetime']
    end_dt = current_dt + timedelta(hours=24)
    
    mask = (df['datetime'] > current_dt) & (df['datetime'] <= end_dt)
    future_df = df.loc[mask].copy()
    
    if len(future_df) > 0:
        future_df['hour'] = future_df['datetime'].dt.hour
        features = future_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
        future_df['pred'] = get_prediction(model, scaler, features) * num_panels
        
        best_slots = future_df.sort_values('pred', ascending=False).head(6)
        
        st.markdown("### ⭐ OPTIMAL TIME SLOTS (NEXT 24H)")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top 3 slots with cards
        cols = st.columns(3)
        for idx, (col, (_, row)) in enumerate(zip(cols, best_slots.head(3).iterrows()), 1):
            with col:
                medal = ["🥇", "🥈", "🥉"][idx-1]
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
                            backdrop-filter: blur(10px);
                            border: 2px solid rgba(0, 255, 136, 0.3);
                            border-radius: 20px;
                            padding: 25px;
                            text-align: center;
                            box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);'>
                    <h1 style='font-size: 3rem; margin: 0;'>{medal}</h1>
                    <h2 style='color: {COLORS['primary']}; margin: 10px 0;'>{row['datetime'].strftime('%H:%M')}</h2>
                    <p style='color: #888; font-size: 0.9rem; margin: 5px 0;'>{row['datetime'].strftime('%d %B')}</p>
                    <h3 style='color: {COLORS['secondary']}; margin: 10px 0;'>{row['pred']:.2f} kW</h3>
                    <p style='color: #888; font-size: 0.85rem;'>☀️ {row['irradiance_W_m2']:.0f} W/m²</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Timeline visualization with heatmap
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Generation Timeline", "Best Slots Heatmap"),
            vertical_spacing=0.12
        )
        
        # Main timeline
        fig.add_trace(go.Scatter(
            x=future_df['datetime'],
            y=future_df['pred'],
            mode='lines',
            name='Generation',
            line=dict(color='rgba(136, 136, 136, 0.4)', width=2),
            fill='tozeroy',
            fillcolor='rgba(136, 136, 136, 0.1)',
            hovertemplate='%{x|%H:%M}<br>%{y:.2f} kW<extra></extra>'
        ), row=1, col=1)
        
        # Highlight best slots
        fig.add_trace(go.Scatter(
            x=best_slots['datetime'],
            y=best_slots['pred'],
            mode='markers',
            name='Optimal',
            marker=dict(
                size=20,
                color=COLORS['primary'],
                symbol='star',
                line=dict(color=COLORS['secondary'], width=3)
            ),
            hovertemplate='<b>OPTIMAL</b><br>%{x|%H:%M}<br>%{y:.2f} kW<extra></extra>'
        ), row=1, col=1)
        
        # Heatmap for quick visual
        hours = future_df['hour'].values
        powers = future_df['pred'].values
        
        # Create heatmap data (reshape for visualization)
        heatmap_data = [[powers[i] if i < len(powers) else 0 for i in range(24)]]
        
    # ✅ NEW / FIXED CODE
fig.add_trace(go.Heatmap(
    z=heatmap_data,
    x=list(range(24)),
    y=['Power'],
    colorscale=[
        [0, '#ff6b6b'],
        [0.5, '#ffd93d'],
        [1, COLORS['primary']]
    ],
    showscale=True,
    hovertemplate='Hour %{x}:00<br>Power: %{z:.2f} kW<extra></extra>',
    colorbar=dict(
        title=dict(text="kW", side="right") # Fixed structure
    )
    ), row=2, col=1)
fig.update_xaxes(title_text="Time", row=1, col=1, gridcolor='#1a1a1a')
fig.update_xaxes(title_text="Hour", row=2, col=1, gridcolor='#1a1a1a')
fig.update_yaxes(title_text="Power (kW)", row=1, col=1, gridcolor='#1a1a1a')
        
fig.update_layout(
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0a0a0a',
            font=dict(color='#888'),
            showlegend=True,
            hovermode='x unified'
        )
        
st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
st.markdown("<br>", unsafe_allow_html=True)
st.info(f"""
        💡 **SMART SCHEDULING TIPS:**
        - Run washing machine at **{best_slots.iloc[0]['datetime'].strftime('%H:%M')}** for maximum solar usage
        - Charge EV during peak hours: **{best_slots.iloc[0]['datetime'].strftime('%H:%M')} - {best_slots.iloc[2]['datetime'].strftime('%H:%M')}**
        - Avoid heavy loads during low generation periods
        - Potential daily savings: **₹{(best_slots.head(3)['pred'].sum() * GRID_RATE):.2f}**
        """)

# --- TAB 5: ROI Analysis ---
with tab5:
    with st.spinner("🔄 Running 30-day simulation..."):
        current_dt = current_row['datetime']
        end_dt = current_dt + timedelta(days=30)
        
        mask = (df['datetime'] >= current_dt) & (df['datetime'] < end_dt)
        sim_df = df.loc[mask].copy()
        
        if len(sim_df) > 0:
            sim_df['hour'] = sim_df['datetime'].dt.hour
            features = sim_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
            sim_df['solar_gen'] = get_prediction(model, scaler, features) * num_panels
            
            bill_A_grid_only = 0
            bill_B_solar_unopt = 0
            bill_C_solar_opt = 0
            
            sim_df['date'] = sim_df['datetime'].dt.date
            
            for date, day_group in sim_df.groupby('date'):
                peak_sun_idx = day_group['solar_gen'].idxmax()
                peak_hour = day_group.loc[peak_sun_idx, 'hour']
                
                for _, row in day_group.iterrows():
                    h = row['hour']
                    solar = row['solar_gen']
                    
                    hourly_load_base = 0
                    hourly_load_base += LOAD_PROFILE["Fans"]["kwh"] * LOAD_PROFILE["Fans"]["qty"]
                    
                    if h >= 22 or h < 6:
                        hourly_load_base += LOAD_PROFILE["AC (1.5 Ton)"]["kwh"] * LOAD_PROFILE["AC (1.5 Ton)"]["qty"]
                    
                    if 18 <= h <= 23:
                        hourly_load_base += LOAD_PROFILE["LEDs"]["kwh"] * LOAD_PROFILE["LEDs"]["qty"]
                    
                    wm_kwh = LOAD_PROFILE["Washing Machine"]["kwh"]
                    
                    # Scenario A
                    load_A = hourly_load_base
                    if h == 20:
                        load_A += wm_kwh
                    bill_A_grid_only += (load_A * GRID_RATE)
                    
                    # Scenario B
                    load_B = hourly_load_base
                    if h == 20:
                        load_B += wm_kwh
                    net_B = max(0, load_B - solar)
                    bill_B_solar_unopt += (net_B * GRID_RATE)
                    
                    # Scenario C
                    load_C = hourly_load_base
                    if h == peak_hour:
                        load_C += wm_kwh
                    net_C = max(0, load_C - solar)
                    bill_C_solar_opt += (net_C * GRID_RATE)
            
            # Top metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔴 GRID ONLY", f"₹{bill_A_grid_only:,.2f}", delta=None, help="No solar, all from grid")
            with col2:
                savings_b = bill_A_grid_only - bill_B_solar_unopt
                st.metric("🟡 SOLAR (UNOPT)", f"₹{bill_B_solar_unopt:,.2f}", delta=f"-₹{savings_b:,.2f}", help="Solar but bad timing")
            with col3:
                savings_c = bill_A_grid_only - bill_C_solar_opt
                st.metric("🟢 SOLAR + AI", f"₹{bill_C_solar_opt:,.2f}", delta=f"-₹{savings_c:,.2f}", help="AI-optimized scheduling")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Comparison chart
            fig_comp = go.Figure()
            
            scenarios = ['Grid Only', 'Solar<br>(Unoptimized)', 'Solar + AI']
            bills = [bill_A_grid_only, bill_B_solar_unopt, bill_C_solar_opt]
            colors_bar = [COLORS['danger'], COLORS['warning'], COLORS['primary']]
            
            for scenario, bill, color in zip(scenarios, bills, colors_bar):
                fig_comp.add_trace(go.Bar(
                    name=scenario,
                    x=[scenario],
                    y=[bill],
                    marker=dict(
                        color=color,
                        line=dict(color='#0a0a0a', width=2)
                    ),
                    text=[f'₹{bill:,.0f}'],
                    textposition='outside',
                    textfont=dict(size=18, color=color, family='Arial Black'),
                    hovertemplate='<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>'
                ))
            
            fig_comp.update_layout(
                title=dict(text="30-DAY COST COMPARISON", font=dict(size=26, color='#888')),
                yaxis_title="Total Cost (₹)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#0a0a0a',
                height=500,
                font={'color': '#888', 'size': 14},
                yaxis=dict(gridcolor='#1a1a1a'),
                showlegend=False
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown("---")
            
            # Savings breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                savings_vs_grid = bill_A_grid_only - bill_C_solar_opt
                annual_savings = savings_vs_grid * 12
                
                fig_save1 = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=savings_vs_grid,
                    number={'prefix': "₹", 'font': {'size': 60, 'color': COLORS['success']}},
                    title={'text': "TOTAL SAVINGS (30 DAYS)", 'font': {'size': 20, 'color': '#888'}},
                    delta={'reference': savings_vs_grid * 0.8, 'relative': False, 'increasing': {'color': COLORS['success']}}
                ))
                fig_save1.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_save1, use_container_width=True)
                
                st.success(f"**📈 Annual Projection:** ₹{annual_savings:,.2f}")
            
            with col2:
                savings_vs_unopt = bill_B_solar_unopt - bill_C_solar_opt
                
                fig_save2 = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=savings_vs_unopt,
                    number={'prefix': "₹", 'font': {'size': 60, 'color': COLORS['warning']}},
                    title={'text': "AI OPTIMIZATION VALUE", 'font': {'size': 20, 'color': '#888'}},
                    delta={'reference': savings_vs_unopt * 0.7, 'relative': False}
                ))
                fig_save2.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_save2, use_container_width=True)
                
                st.info("💡 **Extra savings from smart scheduling**")
            
            st.markdown("---")
            
            # ROI Calculator
            st.markdown("### 📊 ROI CALCULATOR")
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_input, col_output = st.columns([1, 2])
            
            with col_input:
                panel_cost = st.number_input(
                    "💵 Cost per panel (₹)",
                    min_value=10000,
                    max_value=100000,
                    value=25000,
                    step=1000
                )
                
                total_investment = panel_cost * num_panels
                st.metric("💰 Total Investment", f"₹{total_investment:,.0f}")
            
            with col_output:
                monthly_savings = savings_vs_grid
                
                if monthly_savings > 0:
                    payback_months = total_investment / monthly_savings
                    payback_years = payback_months / 12
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("💸 Monthly Saving", f"₹{monthly_savings:,.0f}")
                    with col_b:
                        st.metric("⏱️ Payback Period", f"{payback_years:.1f} years")
                    with col_c:
                        roi_5yr = (monthly_savings * 60) - total_investment
                        st.metric("📈 5-Year Profit", f"₹{roi_5yr:,.0f}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Payback timeline
            if monthly_savings > 0:
                years = list(range(0, min(int(payback_years) + 10, 25)))
                cumulative_savings = [monthly_savings * 12 * y for y in years]
                net_profit = [s - total_investment for s in cumulative_savings]
                
                fig_roi = go.Figure()
                
                # Investment line
                fig_roi.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color='#666',
                    annotation_text="Break-even",
                    annotation_position="right"
                )
                
                # Net profit area
                fig_roi.add_trace(go.Scatter(
                    x=years,
                    y=net_profit,
                    mode='lines+markers',
                    name='Net Profit',
                    line=dict(color=COLORS['primary'], width=4),
                    marker=dict(size=10, color=COLORS['primary'], line=dict(color=COLORS['secondary'], width=2)),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 136, 0.2)',
                    hovertemplate='Year %{x}<br>Profit: ₹%{y:,.0f}<extra></extra>'
                ))
                
                # Breakeven marker
                fig_roi.add_vline(
                    x=payback_years,
                    line_dash="dash",
                    line_color=COLORS['accent'],
                    line_width=3,
                    annotation_text=f"Breakeven: {payback_years:.1f} yrs",
                    annotation_position="top",
                    annotation_font_color=COLORS['accent'],
                    annotation_font_size=16
                )
                
                fig_roi.update_layout(
                    title=dict(text="INVESTMENT TIMELINE", font=dict(size=24, color='#888')),
                    xaxis_title="Years",
                    yaxis_title="Net Profit (₹)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#0a0a0a',
                    height=450,
                    font=dict(color='#888'),
                    hovermode='x unified',
                    xaxis=dict(gridcolor='#1a1a1a'),
                    yaxis=dict(gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#333', zerolinewidth=2)
                )
                
                st.plotly_chart(fig_roi, use_container_width=True)
                
                # Verdict
                if payback_years <= 5:
                    st.success("✅ **EXCELLENT ROI!** System pays for itself in under 5 years. Highly recommended investment.")
                elif payback_years <= 8:
                    st.info("✅ **GOOD ROI.** Solid investment with reasonable payback period.")
                else:
                    st.warning("⚠️ **MODERATE ROI.** Consider increasing panel count or optimizing usage for better returns.")
        else:
            st.warning("⚠️ Insufficient data for 30-day simulation")

# --- Animated Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; padding: 30px;'>
        <h2 style='background: linear-gradient(90deg, #00ff88 0%, #00d4ff 50%, #ff6b9d 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-size: 2.5rem;
                   margin-bottom: 10px;
                   text-shadow: 0 0 40px rgba(0, 255, 136, 0.5);'>
            ☀️ SHEOS AI
        </h2>
        <p style='color: #888; font-size: 1.1rem; margin: 10px 0;'>
            Powered by Machine Learning & Clean Energy 🌍
        </p>
        <p style='color: #666; font-size: 0.95rem;'>
            Data-Driven Insights for Sustainable Living
        </p>
        <br>
        <div style='display: inline-block; padding: 10px 25px; 
                    background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
                    border: 2px solid rgba(0, 255, 136, 0.3);
                    border-radius: 50px;
                    backdrop-filter: blur(10px);'>
            <span style='color: #888; font-size: 0.9rem;'>
                🔋 Optimizing Solar • 💡 Maximizing Savings • 🌱 Minimizing Carbon
            </span>
        </div>
    </div>
    """,unsafe_allow_html=True
)
