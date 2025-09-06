import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# ==============================================================================
# 0. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Global Development Insights",
    page_icon="🌍", 
    layout="wide"
)

# ==============================================================================
# 1. ADVANCED STYLING & UI ENHANCEMENTS
# ==============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp { background-color: #0F1116; }

    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        color: #FFFFFF;
        letter-spacing: -1px;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }

    .st-emotion-cache-z5fcl4 {
        border-radius: 10px;
        padding: 2rem !important;
        background-color: #161B22;
        border: 1px solid #30363d;
    }

    div[data-testid="stMetric"] {
        background-color: #2a313c;
        border: 1px solid #4d5661;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }

    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4CB9E7;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #161B22;
        font-weight: bold;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /*
    FINAL, AGGRESSIVE FIX:
    This targets the tooltip by its accessibility role, which is a much
    more reliable method than the previous attempts.
    */
    div[role="tooltip"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. DATA LOADING AND CACHING (No changes here)
# ==============================================================================
@st.cache_data
def load_and_clean_data(file_path="WorldBank.xlsx"):
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        return None

    data.drop(columns=[
        "Electric power consumption (kWh per capita)", "Infant mortality rate (per 1,000 live births)",
        "Population density (people per sq. km of land area)", "Unemployment (% of total labor force) (modeled ILO estimate)"
    ], inplace=True, errors='ignore')
    
    cols_to_fill = [
        "GDP (USD)", "GDP per capita (USD)", "Life expectancy at birth (years)",
        "Birth rate, crude (per 1,000 people)", "Death rate, crude (per 1,000 people)",
        "Individuals using the Internet (% of population)"
    ]
    for col in cols_to_fill:
        if col in data.columns:
            data[col] = data.sort_values(by=["Country Name", "Year"]).groupby("Country Name")[col].transform(lambda x: x.ffill().bfill())
    
    data.dropna(subset=cols_to_fill + ['IncomeGroup', 'Region', 'Country Code'], inplace=True)
    return data

df = load_and_clean_data()
if df is None:
    st.stop()

# ==============================================================================
# 3. MACHINE LEARNING MODEL TRAINING (No changes here)
# ==============================================================================
@st.cache_resource
def train_model(data):
    ml_cols = ['Individuals using the Internet (% of population)', 'Life expectancy at birth (years)', 'Birth rate, crude (per 1,000 people)', 'GDP per capita (USD)']
    ml_df = data[ml_cols].dropna()
    X = ml_df.drop(columns=['GDP per capita (USD)'])
    y = ml_df['GDP per capita (USD)']
    
    feature_names = X.columns.tolist()

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    return rf, importances, feature_names

model, feature_importances, feature_names = train_model(df)

# ==============================================================================
# 4. SIDEBAR FILTERS
# ==============================================================================
with st.sidebar:
    # REMOVE the old st.image and st.title lines
    st.markdown("## 🌎 Filters") # This is the new, combined title with an emoji
    st.markdown("---")

    min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
    selected_year_range = st.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    sorted_regions = sorted(df['Region'].unique())
    selected_regions = st.multiselect(
        "Select Regions:",
        options=sorted_regions,
        default=sorted_regions
    )

    income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    selected_income_groups = st.multiselect(
        "Select Income Groups:",
        options=income_order,
        default=income_order
    )
    
filtered_df = df[
    (df['Year'] >= selected_year_range[0]) & (df['Year'] <= selected_year_range[1]) &
    (df['Region'].isin(selected_regions)) &
    (df['IncomeGroup'].isin(selected_income_groups))
]

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
    st.stop()

# ==============================================================================
# 5. DASHBOARD LAYOUT
# ==============================================================================
st.markdown("<h1 class='main-header'>Global Development Insights</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='sub-header'>Analyzing Socio-Economic Trends from {selected_year_range[0]} to {selected_year_range[1]}</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🌎  Global Overview", "📊  Comparative Analysis", "🔍  Country Deep Dive", "🤖  GDP Predictor"])

# --- Function to create themed Plotly charts ---
def create_themed_figure(fig, title_text=""):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#E0E0E0",
        title_font_color="#FFD700", # Gold for chart titles
        title_font_size=20,
        title_x=0.5, # Center chart title
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#A0A0E0")),
        xaxis=dict(gridcolor='#3A455A', zerolinecolor='#3A455A'),
        yaxis=dict(gridcolor='#3A455A', zerolinecolor='#3A455A')
    )
    if title_text:
        fig.update_layout(title_text=f"<b>{title_text}</b>")
    return fig

# --- TAB 1: GLOBAL OVERVIEW ---
with tab1:
    with st.container():
        st.subheader("Key Global Metrics")
        latest_year_df = filtered_df[filtered_df['Year'] == selected_year_range[1]]
        
        avg_gdp_capita = latest_year_df['GDP per capita (USD)'].mean()
        avg_life_exp = latest_year_df['Life expectancy at birth (years)'].mean()
        avg_internet_usage = latest_year_df['Individuals using the Internet (% of population)'].mean()
        country_count = filtered_df['Country Name'].nunique()
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Countries Analyzed", value=f"{country_count}")
        kpi2.metric(label="Avg. GDP per Capita (USD)", value=f"{avg_gdp_capita:,.0f}")
        kpi3.metric(label="Avg. Life Expectancy", value=f"{avg_life_exp:.1f} yrs")
        kpi4.metric(label="Avg. Internet Usage", value=f"{avg_internet_usage:.1f}%")
        
    st.markdown("---")
    
    with st.container():
        st.subheader("Global Indicator Map")
        metric_to_map = st.selectbox(
            "Select an indicator to display on the map:",
            ['GDP per capita (USD)', 'Life expectancy at birth (years)', 'Individuals using the Internet (% of population)']
        )
        
        map_df = filtered_df.groupby(['Country Code', 'Country Name'])[metric_to_map].mean().reset_index()
        
        fig_map = px.choropleth(
            map_df,
            locations="Country Code",
            color=metric_to_map,
            hover_name="Country Name",
            color_continuous_scale=px.colors.sequential.Viridis, # FIXED color scale
            title=f"<b>Global Distribution of {metric_to_map}</b>"
        )
        fig_map.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#161B22', landcolor='#2a313c'))
        st.plotly_chart(create_themed_figure(fig_map), use_container_width=True)

# --- TAB 2: COMPARATIVE ANALYSIS ---
with tab2:
    with st.container():
        st.subheader("Trends Over Time by Income Group")
        col1, col2 = st.columns(2)
        with col1:
            gdp_trends = filtered_df.groupby(['Year', 'IncomeGroup'])['GDP per capita (USD)'].mean().reset_index()
            fig1 = px.line(gdp_trends, x='Year', y='GDP per capita (USD)', color='IncomeGroup', title='<b>GDP per Capita Trends</b>', markers=True)
            st.plotly_chart(create_themed_figure(fig1), use_container_width=True)
        with col2:
            internet_trends = filtered_df.groupby(['Year', 'IncomeGroup'])['Individuals using the Internet (% of population)'].mean().reset_index()
            fig2 = px.line(internet_trends, x='Year', y='Individuals using the Internet (% of population)', color='IncomeGroup', title='<b>Internet Usage Trends</b>', markers=True)
            st.plotly_chart(create_themed_figure(fig2), use_container_width=True)

    st.markdown("---")
    
    with st.container():
        st.subheader("Regional & Income Group Averages")
        col3, col4 = st.columns(2)
        with col3:
            region_avg = filtered_df.groupby("Region")["GDP per capita (USD)"].mean().sort_values().reset_index()
            fig3 = px.bar(region_avg, y='Region', x='GDP per capita (USD)', orientation='h', title="<b>Average GDP per Capita by Region</b>")
            st.plotly_chart(create_themed_figure(fig3), use_container_width=True)
        with col4:
            income_avg = filtered_df.groupby("IncomeGroup")["Life expectancy at birth (years)"].mean().reindex(index=income_order).reset_index()
            fig4 = px.bar(income_avg, x='IncomeGroup', y='Life expectancy at birth (years)', title="<b>Average Life Expectancy by Income Group</b>")
            st.plotly_chart(create_themed_figure(fig4), use_container_width=True)

# --- TAB 3: COUNTRY DEEP DIVE ---
with tab3:
    with st.container():
        st.subheader("Explore a Single Country's Journey")
        all_countries = sorted(filtered_df['Country Name'].unique())
        selected_country = st.selectbox("Select a Country:", all_countries)
        
        country_df = filtered_df[filtered_df['Country Name'] == selected_country]
        latest_country_data = country_df.sort_values('Year', ascending=False).iloc[0]
        
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        c_kpi1.metric("Income Group", latest_country_data['IncomeGroup'])
        c_kpi2.metric("Latest GDP per Capita", f"${latest_country_data['GDP per capita (USD)']:,.0f}")
        c_kpi3.metric("Latest Life Expectancy", f"{latest_country_data['Life expectancy at birth (years)']:.1f} yrs")
        
        country_trends = country_df.melt(
            id_vars=['Year'], 
            value_vars=['GDP per capita (USD)', 'Life expectancy at birth (years)', 'Individuals using the Internet (% of population)'],
            var_name='Indicator', value_name='Value'
        )
        
        fig_country = px.line(
            country_trends, x='Year', y='Value', color='Indicator',
            title=f"<b>Development Indicators for {selected_country} Over Time</b>"
        )
        st.plotly_chart(create_themed_figure(fig_country), use_container_width=True)

# --- TAB 4: FUTURE SCENARIOS (Interactive ML with Redesigned Expander) ---
with tab4:
    st.markdown("### 🔮 Predict Economic Futures")
    st.markdown("Adjust key indicators to see their potential impact on GDP per Capita. This model helps understand drivers of economic growth.")

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

    col_scenario1, col_scenario2 = st.columns([0.7, 0.3])
    with col_scenario1:
        st.markdown("#### Adjust Future Indicators:")
        internet_input = st.number_input("Projected Internet Usage (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0, key="internet_future")
        life_exp_input = st.number_input("Projected Life Expectancy (Years)", min_value=40.0, max_value=90.0, value=78.0, step=0.1, key="life_exp_future")
        birth_rate_input = st.number_input("Projected Birth Rate (per 1,000 people)", min_value=5.0, max_value=50.0, value=15.0, step=0.1, key="birth_rate_future")
        
        predict_button = st.button("Simulate Future GDP", type="primary", use_container_width=True)

    if predict_button:
        with st.spinner("Calculating future scenario..."):
            input_data = pd.DataFrame([[internet_input, life_exp_input, birth_rate_input]], columns=feature_names)
            prediction = model.predict(input_data)[0]
            st.session_state.predicted_gdp = prediction
            time.sleep(0.5)

    with col_scenario2:
        st.markdown("#### Predicted Outcome")
        if 'predicted_gdp' not in st.session_state:
            st.session_state.predicted_gdp = 0

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = st.session_state.predicted_gdp,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted GDP per Capita (USD)", 'font': {'size': 20, 'color': '#A0A0A0'}},
            gauge = {
                'axis': {'range': [0, 50000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#FFD700"},
                'bgcolor': "#3A455A",
                'borderwidth': 2,
                'bordercolor': "#A0A0A0",
                'steps': [
                    {'range': [0, 10000], 'color': '#1A212E'},
                    {'range': [10000, 30000], 'color': '#28303E'}],
                'threshold': {
                    'line': {'color': "#4CB9E7", 'width': 4},
                    'thickness': 0.75,
                    'value': 45000}
            },
            number={'font': {'color': "#FFD700", 'size': 50}}
        ))
        
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)
    
    # --- HERE IS THE REDESIGNED SECTION ---
    # We've replaced st.expander with the more modern st.popover
    
    st.markdown("Curious about how the model makes its decisions?")
    
    with st.popover("💡 Explore Model Insights", use_container_width=True):
        st.markdown("The chart below reveals the most influential factors in the model's predictions. A higher score means the factor has a greater impact on the predicted GDP.")
        fig_importance = px.bar(
            feature_importances, 
            x=feature_importances.values, 
            y=feature_importances.index, 
            orientation='h',
            labels={'x': 'Importance Score', 'y': 'Indicator'}
        )
        st.plotly_chart(create_themed_figure(fig_importance, title_text="Key Drivers of GDP Prediction"), use_container_width=True)