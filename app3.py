import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="India EV Market Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved text visibility and contrast
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #F18F01;
        --accent-color: #C73E1D;
        --light-bg: #F7F9FC;
        --dark-text: #222222;
        --text-color: #333333;
        --plot-text: #111111;
    }
    
    body {
        color: var(--dark-text) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background-color: var(--light-bg);
    }
    
    .stApp {
        background-color: var(--light-bg);
    }
    
    p, h1, h2, h3, h4, h5, h6, div, span {
        color: var(--dark-text) !important;
    }
    
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1B6B93;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stSelectbox>div>div>select {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid var(--primary-color);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }
    
    .metric-card h3 {
        color: var(--primary-color);
        font-size: 1.1rem;
        margin-bottom: 10px;
        font-weight: 600;
    }
    
    .metric-card p {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--dark-text);
        margin: 0;
    }
    
    .plot-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .plot-card h3 {
        color: var(--primary-color);
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .section-header {
        color: var(--primary-color);
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 8px;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .info-tooltip {
        background-color: #E3F2FD;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
        border-left: 4px solid var(--primary-color);
    }
    
    .info-tooltip p {
        color: var(--dark-text) !important;
        font-size: 0.95rem;
        line-height: 1.5;
        font-weight: 500;
    }
    
    .tabs .stTab {
        border-radius: 8px !important;
    }
    
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .scenario-card {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 4px solid var(--secondary-color);
    }
    
    .scenario-card h4 {
        color: var(--accent-color);
        margin-top: 0;
        font-weight: 600;
    }
    
    /* Enhanced text visibility for plots */
    .stPlot {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Matplotlib text styling */
    .matplotlib-text {
        color: var(--plot-text) !important;
        font-weight: 500;
    }
    
    /* Table text styling */
    .stDataFrame td, .stDataFrame th {
        color: var(--dark-text) !important;
    }
    
    /* Sidebar text contrast */
    .sidebar .sidebar-content {
        color: var(--dark-text) !important;
    }
    
    /* Footer styling */
    .footer {
        font-size: 0.8rem;
        color: var(--dark-text);
        text-align: center;
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid #eee;
    }
    
    @media (max-width: 768px) {
        .metric-card p {
            font-size: 1.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load data function with updated file paths
@st.cache_data
def load_and_clean_data():
    # Define file paths with your actual directory structure
    base_path = r"C:\Users\Hp\Downloads\archive (5)"
    
    ev_cat_path = base_path + r"\ev_cat_01-24.csv"
    ev_sales_path = base_path + r"\ev_sales_by_makers_and_cat_15-24.csv"
    operational_pc_path = base_path + r"\OperationalPC.csv"
    vehicle_class_path = base_path + r"\Vehicle Class - All.csv"

    # Load Data
    ev_cat_df = pd.read_csv(ev_cat_path)
    ev_sales_df = pd.read_csv(ev_sales_path)
    operational_pc_df = pd.read_csv(operational_pc_path)
    vehicle_class_df = pd.read_csv(vehicle_class_path)

    # Data Cleaning Function
    def clean_data(df):
        df = df.dropna()
        df = df.drop_duplicates()
        df.columns = df.columns.str.upper().str.strip()
        return df

    # Clean all datasets
    ev_cat_df = clean_data(ev_cat_df)
    ev_sales_df = clean_data(ev_sales_df)
    operational_pc_df = clean_data(operational_pc_df)
    vehicle_class_df = clean_data(vehicle_class_df)

    # Feature Engineering
    ev_cat_df['YEAR'] = pd.to_datetime(ev_cat_df['DATE'], errors='coerce').dt.year
    ev_cat_df = ev_cat_df.dropna(subset=['YEAR'])
    ev_cat_df['YEAR'] = ev_cat_df['YEAR'].astype(int)
    
    # Reshape ev_sales_df from wide to long format
    year_cols = [col for col in ev_sales_df.columns if col.isdigit()]
    ev_sales_df = ev_sales_df.melt(
        id_vars=['MAKER', 'CAT'], 
        value_vars=year_cols,
        var_name='YEAR', 
        value_name='SALES'
    )
    ev_sales_df['YEAR'] = ev_sales_df['YEAR'].astype(int)
    
    return ev_cat_df, ev_sales_df, operational_pc_df, vehicle_class_df

# Load data
try:
    ev_cat_df, ev_sales_df, operational_pc_df, vehicle_class_df = load_and_clean_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar with improved styling
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #2E86AB; font-size: 1.5rem; font-weight: 600;">🔌 India EV Dashboard</h1>
        <p style="color: var(--dark-text); font-weight: 500;">Explore the electric vehicle market trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_option = st.radio(
        "Select Analysis",
        ("📊 Market Overview", "🏭 Manufacturer Analysis", "🔮 Sales Prediction", 
         "💡 Market Opportunities", "📈 Adoption Trends", "🚀 Future EV Prediction"),
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: var(--dark-text);">
        <p style="font-weight: 600;">Data Sources:</p>
        <ul>
            <li>EV category data (2001-2024)</li>
            <li>EV sales by manufacturers</li>
            <li>Vehicle class data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.title("🇮🇳 India Electric Vehicle Market Analysis")
st.markdown("""
<div class="info-tooltip">
    <p style="font-weight: 500;">This interactive dashboard provides comprehensive insights into India's rapidly growing electric vehicle market. 
    Explore trends, manufacturer performance, sales predictions, and future opportunities in India's EV sector.</p>
</div>
""", unsafe_allow_html=True)

# Function to style matplotlib plots for better visibility
def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=20, fontweight='bold', color='black', fontsize=14)
    ax.set_xlabel(xlabel, labelpad=10, color='black', fontweight='500')
    ax.set_ylabel(ylabel, labelpad=10, color='black', fontweight='500')
    ax.tick_params(colors='black', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('500')
    plt.grid(True, linestyle='--', alpha=0.3)
    sns.despine()
    return ax

if analysis_option == "📊 Market Overview":
    st.header("Market Overview", anchor="market-overview")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">This section provides a high-level view of India's EV market, including manufacturing trends, 
        market share distribution, and key performance metrics. Use these insights to understand the 
        current state of the EV industry in India.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("<div class='plot-card'><h3>EV Manufacturing Trend</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=ev_cat_df, x='YEAR', y='TWO WHEELER(NT)', label='2W (Non-Transport)', 
                         color='#2E86AB', linewidth=2.5, ax=ax)
            sns.lineplot(data=ev_cat_df, x='YEAR', y='TWO WHEELER(T)', label='2W (Transport)', 
                         color='#F18F01', linewidth=2.5, ax=ax)
            sns.lineplot(data=ev_cat_df, x='YEAR', y='THREE WHEELER(NT)', label='3W (Non-Transport)', 
                         color='#C73E1D', linewidth=2.5, ax=ax)
            ax = style_plot(ax, "EV Manufacturing Trend (2001-2024)", "Year", "Units Manufactured")
            plt.legend(title='Vehicle Type', frameon=False, title_fontsize='12', fontsize=10)
            st.pyplot(fig)
    
    with col2:
        with st.container():
            st.markdown("<div class='plot-card'><h3>Market Share Distribution</h3></div>", unsafe_allow_html=True)
            latest_year = ev_cat_df['YEAR'].max()
            latest_data = ev_cat_df[ev_cat_df['YEAR'] == latest_year].iloc[0]
            
            two_wheelers = latest_data['TWO WHEELER(NT)'] + latest_data['TWO WHEELER(T)']
            three_wheelers = latest_data['THREE WHEELER(NT)'] + latest_data['THREE WHEELER(T)']
            passenger_vehicles = latest_data['LIGHT PASSENGER VEHICLE'] + latest_data['MEDIUM PASSENGER VEHICLE'] + latest_data['HEAVY PASSENGER VEHICLE']
            
            colors = ['#2E86AB', '#F18F01', '#C73E1D']
            explode = (0.05, 0.05, 0.05)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, texts, autotexts = ax.pie(
                [two_wheelers, three_wheelers, passenger_vehicles],
                labels=['Two Wheelers', 'Three Wheelers', 'Passenger Vehicles'],
                autopct='%1.1f%%', 
                startangle=90, 
                colors=colors, 
                explode=explode,
                textprops={'fontsize': 12, 'color': 'black', 'fontweight': '500'}, 
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            plt.setp(autotexts, size=12, weight="bold", color='black')
            ax.set_title(f"Market Share ({latest_year})", pad=20, fontweight='bold', color='black')
            st.pyplot(fig)
    
    # Key metrics with improved layout
    st.subheader("Key Market Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Latest Year Data</h3>
            <p>{latest_year}</p>
            <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Most recent data available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_ev = two_wheelers + three_wheelers + passenger_vehicles
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total EV Units</h3>
            <p>{total_ev:,.0f}</p>
            <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Cumulative production</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Two-Wheeler Dominance</h3>
            <p>{(two_wheelers/total_ev)*100:.1f}%</p>
            <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Of total EV market</p>
        </div>
        """, unsafe_allow_html=True)

elif analysis_option == "🏭 Manufacturer Analysis":
    st.header("Manufacturer Analysis", anchor="manufacturer-analysis")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">This section analyzes EV manufacturers in India, showing sales performance, market share, 
        and trends. Identify top performers and understand competitive dynamics in the EV space.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top Manufacturers
    st.subheader("Top Manufacturers by Sales")
    top_n = st.slider("Select number of top manufacturers to display", 5, 15, 10, 
                      help="Adjust the slider to see more or fewer manufacturers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown(f"<div class='plot-card'><h3>Top {top_n} EV Manufacturers</h3></div>", unsafe_allow_html=True)
            top_manufacturers = ev_sales_df.groupby('MAKER')['SALES'].sum().nlargest(top_n)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_manufacturers.plot(kind='bar', color=sns.color_palette("viridis", top_n), ax=ax)
            ax = style_plot(ax, f"Top {top_n} EV Manufacturers by Total Sales", "Manufacturer", "Total Sales")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    
    with col2:
        # Category composition for top manufacturers
        with st.container():
            st.markdown("<div class='plot-card'><h3>Sales by Vehicle Category</h3></div>", unsafe_allow_html=True)
            category_composition = ev_sales_df[ev_sales_df['MAKER'].isin(top_manufacturers.index)]\
                .groupby(['MAKER', 'CAT'])['SALES'].sum().unstack()
            
            # Filter for EV categories
            ev_categories = [cat for cat in ['2W', '3W'] if cat in category_composition.columns]
            
            if ev_categories:
                fig, ax = plt.subplots(figsize=(10, 6))
                category_composition[ev_categories].plot(kind='bar', stacked=True, 
                                                       color=['#2E86AB', '#F18F01'], ax=ax)
                ax = style_plot(ax, 'EV Sales by Category for Top Manufacturers', 
                               'Manufacturer', 'Sales Volume')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Category', frameon=False, title_fontsize='12', fontsize=10)
                st.pyplot(fig)
            else:
                st.warning("No EV category data available for selected manufacturers")
    
    # Manufacturer trends
    st.subheader("Manufacturer Sales Trends")
    selected_maker = st.selectbox(
        "Select manufacturer to view trend",
        ev_sales_df['MAKER'].unique(),
        index=0,
        help="Select a manufacturer to see their sales trend over time"
    )
    
    with st.container():
        st.markdown(f"<div class='plot-card'><h3>{selected_maker} Sales Trend</h3></div>", unsafe_allow_html=True)
        maker_trend = ev_sales_df[ev_sales_df['MAKER'] == selected_maker].groupby('YEAR')['SALES'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        maker_trend.plot(kind='line', marker='o', color='#2E86AB', linewidth=2.5, ax=ax)
        ax = style_plot(ax, f"Sales Trend for {selected_maker}", "Year", "Annual Sales")
        st.pyplot(fig)
    
    # Clustering analysis
    if st.checkbox("Show Manufacturer Clustering Analysis", help="Cluster manufacturers based on their sales patterns"):
        st.subheader("Manufacturer Clustering")
        st.markdown("""
        <div class="info-tooltip">
            <p style="font-weight: 500;">This analysis groups manufacturers into clusters based on their sales patterns across different 
            vehicle categories. Similar manufacturers are grouped together, helping identify competitive 
            groups in the market.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare manufacturer data
        manufacturer_data = ev_sales_df.pivot_table(
            index='MAKER',
            columns='CAT',
            values='SALES',
            aggfunc='sum'
        ).fillna(0)
        
        # Focus on electric vehicle categories
        ev_categories = ['2W', '3W']
        available_ev_categories = [cat for cat in ev_categories if cat in manufacturer_data.columns]
        
        if len(available_ev_categories) >= 2:
            manufacturer_data = manufacturer_data[available_ev_categories]
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(manufacturer_data)
            
            # Elbow method
            with st.container():
                st.markdown("<div class='plot-card'><h3>Determining Optimal Number of Clusters</h3></div>", unsafe_allow_html=True)
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                    kmeans.fit(scaled_data)
                    wcss.append(kmeans.inertia_)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.plot(range(1, 11), wcss, marker='o', color='#2E86AB', linewidth=2.5)
                ax = style_plot(ax, 'Elbow Method: Optimal Number of Clusters', 
                               'Number of Clusters', 'Within-Cluster Sum of Squares (WCSS)')
                st.pyplot(fig)
            
            # Apply K-Means clustering
            n_clusters = st.slider("Select number of clusters", 2, 5, 3, 
                                 help="Choose the number of clusters based on the elbow point")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Visualize clusters
            with st.container():
                st.markdown("<div class='plot-card'><h3>Manufacturer Clusters</h3></div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 7))
                scatter = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], 
                                   c=clusters, cmap='viridis', alpha=0.8, 
                                   s=150, edgecolor='k', linewidth=0.5)
                
                # Plot cluster centers
                ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                          s=300, c='red', marker='X', label='Cluster Centers')
                
                # Annotate points with manufacturer names
                for i, maker in enumerate(manufacturer_data.index):
                    ax.annotate(maker, (scaled_data[i, 0], scaled_data[i, 1]), 
                               textcoords="offset points", xytext=(0,5), ha='center', 
                               fontsize=8, color='black', fontweight='500')
                
                ax = style_plot(ax, "Manufacturer Clusters by EV Sales Performance", 
                               f"Standardized {available_ev_categories[0]} Sales", 
                               f"Standardized {available_ev_categories[1]} Sales")
                ax.legend(frameon=False, fontsize=10)
                plt.colorbar(scatter, label='Cluster')
                st.pyplot(fig)
            
            # Cluster profiles
            st.subheader("Cluster Profiles")
            manufacturer_data['CLUSTER'] = clusters
            cluster_profile = manufacturer_data.groupby('CLUSTER').mean()
            
            # Style the dataframe
            def color_cluster(val):
                color = plt.cm.viridis(val/cluster_profile.max().max())
                return f'background-color: rgba({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f},{color[3]*0.3})'
            
            st.dataframe(cluster_profile.style.applymap(color_cluster).format("{:.2f}"))
            
        else:
            st.warning("Need at least 2 EV categories for clustering analysis")

elif analysis_option == "🔮 Sales Prediction":
    st.header("Sales Prediction Models")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">This section allows you to forecast future EV sales using different machine learning models. 
        Compare model performance and see predictions for various vehicle categories.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Select target variable
    target_options = {
        'Two Wheeler (Non-Transport)': 'TWO WHEELER(NT)',
        'Two Wheeler (Transport)': 'TWO WHEELER(T)',
        'Three Wheeler (Non-Transport)': 'THREE WHEELER(NT)'
    }
    target_var = st.selectbox("Select target variable for prediction", list(target_options.keys()),
                            help="Choose which vehicle category to predict")
    
    # Model selection
    models_to_run = st.multiselect(
        "Select models to run",
        ['Linear Regression', 'Ridge Regression', 'Random Forest', 'ARIMA'],
        default=['Linear Regression', 'Random Forest'],
        help="Select which prediction models to compare"
    )
    
    if st.button("Run Prediction Models", key="predict_button"):
        st.subheader("Model Performance")
        st.markdown("""
        <div class="info-tooltip">
            <p style="font-weight: 500;">The table below shows the performance metrics for each selected model. 
            R² score indicates how well the model explains the variance in the data (closer to 1 is better), 
            while RMSE measures prediction error (lower is better).</p>
        </div>
        """, unsafe_allow_html=True)
        
        results = []
        predictions = {}
        
        # Prepare data
        X = ev_cat_df[['YEAR']]
        y = ev_cat_df[target_options[target_var]]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if 'Linear Regression' in models_to_run:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_r2 = r2_score(y_test, lr_pred)
            results.append({
                'Model': 'Linear Regression',
                'R2 Score': lr_r2,
                'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred))
            })
            predictions['Linear Regression'] = lr_pred
        
        if 'Ridge Regression' in models_to_run:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            ridge_pred = ridge.predict(X_test)
            ridge_r2 = r2_score(y_test, ridge_pred)
            results.append({
                'Model': 'Ridge Regression',
                'R2 Score': ridge_r2,
                'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred))
            })
            predictions['Ridge Regression'] = ridge_pred
        
        if 'Random Forest' in models_to_run:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_r2 = r2_score(y_test, rf_pred)
            results.append({
                'Model': 'Random Forest',
                'R2 Score': rf_r2,
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred))
            })
            predictions['Random Forest'] = rf_pred
        
        if 'ARIMA' in models_to_run:
            try:
                y_values = y.values
                train_size = int(len(y_values) * 0.8)
                train, test = y_values[:train_size], y_values[train_size:]
                
                history = [x for x in train]
                arima_predictions = []
                
                for t in range(len(test)):
                    model = ARIMA(history, order=(5,1,0))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    arima_predictions.append(yhat)
                    history.append(test[t])
                
                arima_rmse = np.sqrt(mean_squared_error(test, arima_predictions))
                results.append({
                    'Model': 'ARIMA',
                    'R2 Score': np.nan,
                    'RMSE': arima_rmse
                })
                predictions['ARIMA'] = arima_predictions
            except Exception as e:
                st.error(f"ARIMA modeling failed: {str(e)}")
        
        # Display results with styling
        results_df = pd.DataFrame(results)
        
        def highlight_max(s, props=''):
            return np.where(s == s.max(), props, '')
        
        styled_df = results_df.style\
            .format({
                'R2 Score': '{:.3f}',
                'RMSE': '{:.1f}'
            })\
            .apply(highlight_max, subset=['R2 Score'], props='background-color: #E3F2FD; font-weight: bold;')\
            .apply(highlight_max, subset=['RMSE'], props='background-color: #FFEBEE; font-weight: bold;')
        
        st.dataframe(styled_df)
        
        # Plot predictions
        st.subheader("Actual vs Predicted Values")
        st.markdown("""
        <div class="info-tooltip">
            <p style="font-weight: 500;">The chart below compares actual sales values with predictions from each model. 
            This visualization helps assess how well each model fits the actual data patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='plot-card'><h3>Model Predictions Comparison</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_test.values, label='Actual', color='#2E86AB', linewidth=3, alpha=0.8)
            
            colors = ['#F18F01', '#C73E1D', '#6A4C93', '#28A745']
            for i, (model_name, pred) in enumerate(predictions.items()):
                if model_name == 'ARIMA':
                    ax.plot(range(len(y_test)-len(pred), len(y_test)), pred, label=model_name, 
                           color=colors[i], linestyle='--', linewidth=2)
                else:
                    ax.plot(pred, label=model_name, color=colors[i], linestyle='--', linewidth=2)
            
            ax = style_plot(ax, f"Actual vs Predicted {target_var} Sales", "Test Samples", target_var)
            ax.legend(frameon=False, fontsize=10)
            st.pyplot(fig)

elif analysis_option == "💡 Market Opportunities":
    st.header("Market Opportunity Analysis")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">This section identifies growth opportunities in India's EV market, analyzing growth rates, 
        market concentration, and the impact of government policies on EV adoption.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CAGR calculation
    st.subheader("Market Growth Analysis")
    first_year = ev_cat_df['YEAR'].min()
    last_year = ev_cat_df['YEAR'].max()
    periods = last_year - first_year
    
    initial_sales = ev_cat_df[ev_cat_df['YEAR'] == first_year]['TWO WHEELER(NT)'].sum()
    final_sales = ev_cat_df[ev_cat_df['YEAR'] == last_year]['TWO WHEELER(NT)'].sum()
    
    cagr = (final_sales / initial_sales) ** (1/periods) - 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>CAGR ({first_year}-{last_year})</h3>
            <p>{cagr*100:.2f}%</p>
            <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Compound Annual Growth Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Market projection
        projection_years = st.slider("Projection period (years)", 1, 10, 5, 
                                   help="Select how many years into the future to project growth")
        projected_market = final_sales * (1 + cagr) ** projection_years
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Projected Market Size</h3>
            <p>{projected_market:,.0f} units</p>
            <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Estimated in {projection_years} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Market concentration
    st.subheader("Market Concentration")
    top_5_share = ev_sales_df[ev_sales_df['CAT'] == '2W'].groupby('MAKER')['SALES'].sum().nlargest(5).sum()
    total_2w_sales = ev_sales_df[ev_sales_df['CAT'] == '2W']['SALES'].sum()
    concentration = (top_5_share / total_2w_sales) * 100
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>Top 5 Manufacturers Market Share</h3>
        <p>{concentration:.1f}%</p>
        <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Of total two-wheeler EV market</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Policy impact
    st.subheader("Policy Impact Analysis")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">The chart below shows how key government policies have impacted EV adoption in India. 
        Vertical lines mark important policy announcements that influenced market growth.</p>
    </div>
    """, unsafe_allow_html=True)
    
    policy_years = {2015: 'FAME I Launch', 2019: 'FAME II Launch', 2021: 'PLI Scheme'}
    
    with st.container():
        st.markdown("<div class='plot-card'><h3>EV Sales with Policy Interventions</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=ev_cat_df, x='YEAR', y='TWO WHEELER(NT)', color='#2E86AB', linewidth=2.5, ax=ax)
        
        for year, label in policy_years.items():
            ax.axvline(x=year, color='#C73E1D', linestyle='--', alpha=0.7)
            ax.text(year, ev_cat_df['TWO WHEELER(NT)'].max()*0.8, label, rotation=90, 
                    verticalalignment='center', color='black', fontweight='500', fontsize=10)
        
        ax = style_plot(ax, 'EV Sales with Key Policy Interventions', 'Year', 'Two-Wheeler Sales')
        st.pyplot(fig)

elif analysis_option == "📈 Adoption Trends":
    st.header("Technology Adoption Trends")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">This section analyzes the adoption curve of EV technology in India, showing how different 
        vehicle categories have grown over time and identifying key adoption milestones.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Adoption curve
    st.subheader("EV Adoption Curve")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">The adoption curve shows cumulative EV production over time, helping identify when 
        different vehicle categories reached significant market penetration.</p>
    </div>
    """, unsafe_allow_html=True)
    
    adoption = ev_cat_df.groupby('YEAR')[['TWO WHEELER(NT)', 'THREE WHEELER(NT)']].sum().cumsum()
    
    with st.container():
        st.markdown("<div class='plot-card'><h3>EV Technology Adoption Curve</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2E86AB', '#F18F01']
        for i, col in enumerate(adoption.columns):
            sns.lineplot(x=adoption.index, y=adoption[col], label=col.replace('(NT)', '').title(), 
                         color=colors[i], linewidth=2.5, ax=ax)
        
        ax = style_plot(ax, 'Cumulative EV Production Over Time', 'Year', 'Cumulative Units Produced')
        plt.legend(title='Vehicle Type', frameon=False, title_fontsize='12', fontsize=10)
        st.pyplot(fig)
    
    # Growth phases
    st.subheader("Market Growth Phases")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">This analysis identifies different phases of market growth based on annual production increases.</p>
    </div>
    """, unsafe_allow_html=True)
    
    growth = ev_cat_df.groupby('YEAR')[['TWO WHEELER(NT)', 'THREE WHEELER(NT)']].sum().diff().dropna()
    
    with st.container():
        st.markdown("<div class='plot-card'><h3>Annual Production Growth</h3></div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, col in enumerate(growth.columns):
            sns.lineplot(x=growth.index, y=growth[col], label=col.replace('(NT)', '').title(), 
                         color=colors[i], linewidth=2.5, ax=ax)
        
        # Identify growth phases
        ax.axvspan(2001, 2010, color='gray', alpha=0.1, label='Early Stage')
        ax.axvspan(2011, 2018, color='green', alpha=0.1, label='Growth Stage')
        ax.axvspan(2019, 2024, color='blue', alpha=0.1, label='Acceleration Stage')
        
        ax = style_plot(ax, 'Annual EV Production Growth by Phase', 'Year', 'Year-over-Year Growth')
        plt.legend(title='Growth Phase', frameon=False, fontsize=10)
        st.pyplot(fig)

elif analysis_option == "🚀 Future EV Prediction":
    st.header("Future EV Market Prediction")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">This section provides future projections for India's EV market under different scenarios, 
        helping stakeholders plan for various potential futures.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Scenario analysis
    st.subheader("Market Projection Scenarios")
    st.markdown("""
    <div class="info-tooltip">
        <p style="font-weight: 500;">Explore how different policy and economic conditions might affect EV adoption in India.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get latest year data
    latest_year = ev_cat_df['YEAR'].max()
    latest_2w = ev_cat_df[ev_cat_df['YEAR'] == latest_year]['TWO WHEELER(NT)'].values[0]
    
    # Scenario parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='scenario-card'><h4>Base Scenario</h4></div>", unsafe_allow_html=True)
        base_growth = st.slider("Base growth rate (%)", 5.0, 30.0, 15.0, 0.5, key='base_growth')
    
    with col2:
        st.markdown("<div class='scenario-card'><h4>Optimistic Scenario</h4></div>", unsafe_allow_html=True)
        optimistic_growth = st.slider("Optimistic growth rate (%)", 15.0, 50.0, 25.0, 0.5, key='opt_growth')
    
    with col3:
        st.markdown("<div class='scenario-card'><h4>Pessimistic Scenario</h4></div>", unsafe_allow_html=True)
        pessimistic_growth = st.slider("Pessimistic growth rate (%)", 0.0, 20.0, 8.0, 0.5, key='pess_growth')
    
    projection_years = st.slider("Years to project", 1, 15, 10, help="Select number of years for projection")
    
    if st.button("Generate Projections", key="project_button"):
        st.subheader("Market Projection Results")
        
        # Create projection data
        years = list(range(latest_year, latest_year + projection_years + 1))
        base = [latest_2w * (1 + base_growth/100)**i for i in range(projection_years + 1)]
        optimistic = [latest_2w * (1 + optimistic_growth/100)**i for i in range(projection_years + 1)]
        pessimistic = [latest_2w * (1 + pessimistic_growth/100)**i for i in range(projection_years + 1)]
        
        # Plot projections
        with st.container():
            st.markdown("<div class='plot-card'><h3>EV Market Projections</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Historical data
            hist_data = ev_cat_df[ev_cat_df['YEAR'] >= 2010][['YEAR', 'TWO WHEELER(NT)']]
            sns.lineplot(x=hist_data['YEAR'], y=hist_data['TWO WHEELER(NT)'], 
                        label='Historical', color='#2E86AB', linewidth=3, ax=ax)
            
            # Projections
            sns.lineplot(x=years, y=base, label=f'Base ({base_growth}%)', 
                        color='#F18F01', linestyle='--', linewidth=2.5, ax=ax)
            sns.lineplot(x=years, y=optimistic, label=f'Optimistic ({optimistic_growth}%)', 
                        color='#28A745', linestyle='--', linewidth=2.5, ax=ax)
            sns.lineplot(x=years, y=pessimistic, label=f'Pessimistic ({pessimistic_growth}%)', 
                        color='#C73E1D', linestyle='--', linewidth=2.5, ax=ax)
            
            # Current year marker
            ax.axvline(x=latest_year, color='gray', linestyle=':', alpha=0.7)
            ax.text(latest_year, ax.get_ylim()[1]*0.9, 'Current', rotation=90, 
                   verticalalignment='top', color='black', fontweight='500')
            
            ax = style_plot(ax, 'Two-Wheeler EV Market Projections', 'Year', 'Units Produced')
            plt.legend(frameon=False, fontsize=10)
            st.pyplot(fig)
        
        # Display projection metrics
        st.subheader("Projection Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Base Scenario</h3>
                <p>{base[-1]:,.0f}</p>
                <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Projected units in {years[-1]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Optimistic Scenario</h3>
                <p>{optimistic[-1]:,.0f}</p>
                <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Projected units in {years[-1]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Pessimistic Scenario</h3>
                <p>{pessimistic[-1]:,.0f}</p>
                <p style="font-size: 0.9rem; color: var(--dark-text); font-weight: 500;">Projected units in {years[-1]}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>India EV Market Analysis Dashboard • Data through 2024 • Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)