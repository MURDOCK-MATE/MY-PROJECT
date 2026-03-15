"""
Live Data Visualization Dashboard - Backend
============================================
A Flask application that fetches live data from multiple sources and generates
visualizations using Seaborn and Matplotlib.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from io import BytesIO
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Set style for better-looking graphs
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ==============================================================================
# POPULAR EXAMPLES FOR EACH DATASET
# ==============================================================================

DATASET_EXAMPLES = {
    'stock': [
        {'value': 'AAPL', 'label': 'Apple Inc. (AAPL)'},
        {'value': 'TSLA', 'label': 'Tesla (TSLA)'},
        {'value': 'GOOGL', 'label': 'Google (GOOGL)'},
        {'value': 'MSFT', 'label': 'Microsoft (MSFT)'},
        {'value': 'AMZN', 'label': 'Amazon (AMZN)'},
        {'value': 'META', 'label': 'Meta/Facebook (META)'},
        {'value': 'NVDA', 'label': 'NVIDIA (NVDA)'},
        {'value': 'JPM', 'label': 'JP Morgan (JPM)'},
    ],
    'weather': [
        {'value': 'London', 'label': 'London, UK'},
        {'value': 'New York', 'label': 'New York, USA'},
        {'value': 'Tokyo', 'label': 'Tokyo, Japan'},
        {'value': 'Paris', 'label': 'Paris, France'},
        {'value': 'Delhi', 'label': 'Delhi, India'},
        {'value': 'Mumbai', 'label': 'Mumbai, India'},
        {'value': 'Dubai', 'label': 'Dubai, UAE'},
        {'value': 'Sydney', 'label': 'Sydney, Australia'},
        {'value': 'Singapore', 'label': 'Singapore'},
        {'value': 'Berlin', 'label': 'Berlin, Germany'},
    ],
    'air_quality': [
        {'value': 'Delhi', 'label': 'Delhi, India'},
        {'value': 'Beijing', 'label': 'Beijing, China'},
        {'value': 'London', 'label': 'London, UK'},
        {'value': 'Los Angeles', 'label': 'Los Angeles, USA'},
        {'value': 'Mumbai', 'label': 'Mumbai, India'},
        {'value': 'Paris', 'label': 'Paris, France'},
        {'value': 'Mexico City', 'label': 'Mexico City, Mexico'},
        {'value': 'Bangkok', 'label': 'Bangkok, Thailand'},
    ],
    'crypto': [
        {'value': 'bitcoin', 'label': 'Bitcoin (BTC)'},
        {'value': 'ethereum', 'label': 'Ethereum (ETH)'},
        {'value': 'cardano', 'label': 'Cardano (ADA)'},
        {'value': 'dogecoin', 'label': 'Dogecoin (DOGE)'},
        {'value': 'solana', 'label': 'Solana (SOL)'},
        {'value': 'polkadot', 'label': 'Polkadot (DOT)'},
        {'value': 'ripple', 'label': 'Ripple (XRP)'},
        {'value': 'litecoin', 'label': 'Litecoin (LTC)'},
    ]
}

# ==============================================================================
# DATA FETCHING FUNCTIONS
# ==============================================================================

def fetch_stock_data(symbol, period='1mo'):
    """
    Fetch stock market data from Yahoo Finance
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            return None
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Select relevant columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None


def fetch_weather_data(city):
    try:
        API_KEY = '8155fc6413c93b99ff4e459752575f84'

        city = city.strip()

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        df = pd.DataFrame([{
            'City': data['name'],
            'Temperature': data['main']['temp'],
            'Humidity': data['main']['humidity'],
            'WindSpeed': data['wind']['speed'],
            'Pressure': data['main']['pressure'],
            'Weather': data['weather'][0]['description'],
            'DateTime': datetime.fromtimestamp(data['dt'])
        }])

        return df

    except Exception as e:
        print("Weather Error:", e)
        return None


def fetch_air_quality_data(city):
    """
    Fetch air quality data from World Air Quality Index (WAQI)
    
    Args:
        city (str): City name
    
    Returns:
        pd.DataFrame: Air quality measurements
    """
    try:
        url = f'https://api.waqi.info/feed/{city}/?token=demo'
        
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"Air Quality API Error: Status {response.status_code}")
            return None
        
        data = response.json()
        
        if data.get('status') != 'ok':
            print(f"No air quality data found for city: {city}")
            return None
        
        result = data['data']
        
        records = []
        
        records.append({
            'Parameter': 'AQI',
            'Value': result.get('aqi', 0),
            'Unit': 'AQI',
            'Location': result.get('city', {}).get('name', city),
            'DateTime': result.get('time', {}).get('s', 'N/A')
        })
        
        iaqi = result.get('iaqi', {})
        
        pollutant_map = {
            'pm25': ('PM2.5', 'µg/m³'),
            'pm10': ('PM10', 'µg/m³'),
            'o3': ('Ozone', 'µg/m³'),
            'no2': ('NO2', 'µg/m³'),
            'so2': ('SO2', 'µg/m³'),
            'co': ('CO', 'µg/m³'),
            't': ('Temperature', '°C'),
            'h': ('Humidity', '%'),
            'p': ('Pressure', 'hPa'),
            'w': ('Wind Speed', 'km/h')
        }
        
        for key, (name, unit) in pollutant_map.items():
            if key in iaqi:
                records.append({
                    'Parameter': name,
                    'Value': iaqi[key].get('v', 0),
                    'Unit': unit,
                    'Location': result.get('city', {}).get('name', city),
                    'DateTime': result.get('time', {}).get('s', 'N/A')
                })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        
        df_wide = df.pivot_table(
            index=['Location', 'DateTime'],
            columns='Parameter',
            values='Value',
            aggfunc='first'
        ).reset_index()
        
        return df_wide
        
    except Exception as e:
        print(f"Error fetching air quality data: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_crypto_data(coin_id):
    """
    Fetch cryptocurrency data from CoinGecko
    
    Args:
        coin_id (str): Coin identifier (e.g., 'bitcoin', 'ethereum')
    
    Returns:
        pd.DataFrame: Cryptocurrency price and volume data
    """
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30'
        
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        prices = data['prices']
        volumes = data['total_volumes']
        market_caps = data['market_caps']
        
        records = []
        for i in range(len(prices)):
            records.append({
                'DateTime': datetime.fromtimestamp(prices[i][0] / 1000).strftime('%Y-%m-%d'),
                'Price': prices[i][1],
                'Volume': volumes[i][1],
                'MarketCap': market_caps[i][1]
            })
        
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        return None


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_plot(df, plot_type, x_col, y_col, hue_col=None):
    """
    Create a plot based on specified parameters
    
    Args:
        df (pd.DataFrame): Data to plot
        plot_type (str): Type of plot ('line', 'bar', 'scatter', 'histogram', 'box', 'heatmap')
        x_col (str): Column for x-axis
        y_col (str): Column for y-axis
        hue_col (str, optional): Column for color grouping
    
    Returns:
        str: Base64 encoded image
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if plot_type == 'line':
            if hue_col and hue_col in df.columns:
                sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker='o', ax=ax)
            else:
                sns.lineplot(data=df, x=x_col, y=y_col, marker='o', ax=ax)
        
        elif plot_type == 'bar':
            if hue_col and hue_col in df.columns:
                sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        
        elif plot_type == 'scatter':
            if hue_col and hue_col in df.columns:
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=100, ax=ax)
            else:
                sns.scatterplot(data=df, x=x_col, y=y_col, s=100, ax=ax)
        
        elif plot_type == 'histogram':
            if hue_col and hue_col in df.columns:
                sns.histplot(data=df, x=x_col, hue=hue_col, kde=True, ax=ax)
            else:
                sns.histplot(data=df, x=x_col, kde=True, ax=ax)
        
        elif plot_type == 'box':
            if hue_col and hue_col in df.columns:
                sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                if x_col and y_col:
                    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                else:
                    sns.boxplot(data=df, ax=ax)
        
        elif plot_type == 'heatmap':
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            plt.title('Correlation Heatmap')
        
        if plot_type != 'heatmap':
            plt.xlabel(x_col, fontsize=12, fontweight='bold')
            plt.ylabel(y_col, fontsize=12, fontweight='bold')
            plt.title(f'{plot_type.capitalize()} Plot: {y_col} vs {x_col}', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        plt.close()
        return None


# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')


@app.route('/get_examples', methods=['POST'])
def get_examples():
    """Return popular examples for a given dataset type"""
    try:
        data = request.json
        dataset_type = data.get('dataset')
        
        if dataset_type in DATASET_EXAMPLES:
            return jsonify({
                'examples': DATASET_EXAMPLES[dataset_type]
            })
        else:
            return jsonify({'examples': []})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_columns', methods=['POST'])
def get_columns():
    """Fetch data and return available columns for dropdown population"""
    try:
        data = request.json
        dataset_type = data.get('dataset')
        input_value = data.get('input')
        
        df = None
        
        if dataset_type == 'stock':
            df = fetch_stock_data(input_value.upper())
        elif dataset_type == 'weather':
            df = fetch_weather_data(input_value)
        elif dataset_type == 'air_quality':
            df = fetch_air_quality_data(input_value)
        elif dataset_type == 'crypto':
            df = fetch_crypto_data(input_value.lower())
        
        if df is None or df.empty:
            suggestions = [ex['label'] for ex in DATASET_EXAMPLES.get(dataset_type, [])[:3]]
            error_msg = f'No data found for "{input_value}". Try one of these: {", ".join(suggestions)}'
            return jsonify({'error': error_msg}), 404
        
        columns = df.columns.tolist()
        
        return jsonify({
            'columns': columns,
            'preview': df.head(5).to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    """Generate and return a plot based on user selections"""
    try:
        data = request.json
        
        dataset_type = data.get('dataset')
        input_value = data.get('input')
        plot_type = data.get('plot_type')
        x_col = data.get('x_axis')
        y_col = data.get('y_axis')
        hue_col = data.get('hue')
        
        df = None
        if dataset_type == 'stock':
            df = fetch_stock_data(input_value.upper())
        elif dataset_type == 'weather':
            df = fetch_weather_data(input_value)
        elif dataset_type == 'air_quality':
            df = fetch_air_quality_data(input_value)
        elif dataset_type == 'crypto':
            df = fetch_crypto_data(input_value.lower())
        
        if df is None or df.empty:
            return jsonify({'error': 'Failed to fetch data'}), 404
        
        image_base64 = create_plot(df, plot_type, x_col, y_col, hue_col)
        
        if image_base64 is None:
            return jsonify({'error': 'Failed to generate plot'}), 500
        
        return jsonify({
            'image': image_base64,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==============================================================================
# AWP (ANTENNA & WAVE PROPAGATION) - GTU DASHBOARD
# ==============================================================================

def dipole_pattern(theta, L_over_lambda):
    """
    Compute the normalised E-field radiation pattern for a centre-fed dipole.

    Uses the standard formula:
        E(θ) = |[cos(β·L/2·cosθ) − cos(β·L/2)] / sinθ|
    where β = 2π/λ, so  β·L/2 = π·(L/λ).

    Args:
        theta (np.ndarray): Angles in radians, shape (N,)
        L_over_lambda (float): Dipole length normalised to wavelength

    Returns:
        np.ndarray: Normalised pattern values in [0, 1]
    """
    beta_L_half = np.pi * L_over_lambda          # β·L/2
    sin_theta = np.sin(theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.cos(beta_L_half * np.cos(theta)) - np.cos(beta_L_half)
        E = np.where(np.abs(sin_theta) < 1e-10, 0.0,
                     np.abs(numerator / sin_theta))
    E_max = np.max(E)
    return E / E_max if E_max > 0 else E


def generate_awp_dashboard():
    """
    Generate a 6-panel polar radiation-pattern dashboard covering the GTU AWP
    syllabus:

    Panel 1 – Short Dipole (Hertzian)  : E(θ) = |sinθ|
    Panel 2 – Half-Wave Dipole (λ/2)   : standard formula, L/λ = 0.5
    Panel 3 – Full-Wave Dipole (λ)     : standard formula, L/λ = 1.0
    Panel 4 – 3λ/2 Dipole              : standard formula, L/λ = 1.5
    Panel 5 – λ/4 Monopole (grounded)  : upper hemisphere only
    Panel 6 – 2-Element Broadside Array: element × array-factor, d = λ/2

    Returns:
        str: Base64-encoded PNG image of the 6-panel figure.
    """
    theta = np.linspace(0, 2 * np.pi, 1000)

    # ── Panel configurations ───────────────────────────────────────────────
    # (pattern_array, color, title)
    panels = []

    # 1. Short / Hertzian dipole  E(θ) = |sinθ|
    E1 = np.abs(np.sin(theta))
    panels.append((E1, '#3b82f6',
                   'Short Dipole (Hertzian)\nL ≪ λ  →  E(θ) = |sin θ|'))

    # 2. Half-wave dipole  L = λ/2
    E2 = dipole_pattern(theta, 0.5)
    panels.append((E2, '#10b981',
                   'Half-Wave Dipole\nL = λ/2'))

    # 3. Full-wave dipole  L = λ
    E3 = dipole_pattern(theta, 1.0)
    panels.append((E3, '#ef4444',
                   'Full-Wave Dipole\nL = λ'))

    # 4. 3λ/2 dipole
    E4 = dipole_pattern(theta, 1.5)
    panels.append((E4, '#8b5cf6',
                   '3λ/2 Dipole\nL = 3λ/2'))

    # 5. Quarter-wave monopole above ground plane
    #    Pattern = half-wave dipole but zero below the horizon (θ > π/2 and < 3π/2)
    E5_full = dipole_pattern(theta, 0.5)
    E5 = np.where((theta <= np.pi / 2) | (theta >= 3 * np.pi / 2), E5_full, 0.0)
    panels.append((E5, '#f59e0b',
                   'λ/4 Monopole (Grounded)\nUpper hemisphere only'))

    # 6. 2-element broadside array (d = λ/2, in-phase excitation)
    #    Array Factor: AF = |cos(π/2 · cosθ)|
    #    Element pattern: short dipole  |sinθ|
    AF = np.abs(np.cos(np.pi / 2 * np.cos(theta)))
    E6 = np.abs(np.sin(theta)) * AF
    E6_max = np.max(E6)
    E6 = E6 / E6_max if E6_max > 0 else E6
    panels.append((E6, '#06b6d4',
                   '2-Element Broadside Array\nd = λ/2, in-phase'))

    # ── Build figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                             subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle(
        'Antenna Radiation Patterns  –  AWP Dashboard (GTU)',
        fontsize=15, fontweight='bold', color='white', y=1.01
    )

    for ax, (E, color, title) in zip(axes.flat, panels):
        ax.set_facecolor('#1e293b')
        ax.plot(theta, E, color=color, linewidth=2)
        ax.fill(theta, E, alpha=0.25, color=color)
        ax.set_title(title, fontsize=9, fontweight='bold',
                     color='white', pad=12)
        ax.set_theta_zero_location('N')   # 0° at top (antenna axis)
        ax.set_theta_direction(-1)        # clockwise
        ax.grid(True, color='gray', alpha=0.3)
        ax.tick_params(colors='gray', labelsize=7)
        ax.set_yticklabels([])
        ax.set_rlabel_position(45)
        # Degree labels
        ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                          labels=['0°', '30°', '60°', '90°', '120°', '150°',
                                  '180°', '210°', '240°', '270°', '300°', '330°'],
                          fontsize=6, color='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    plt.tight_layout(pad=2.0)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64


@app.route('/awp')
def awp():
    """Render the AWP (Antenna & Wave Propagation) radiation pattern dashboard."""
    return render_template('awp.html')


@app.route('/awp_plot')
def awp_plot():
    """Return the 6-panel AWP radiation-pattern dashboard as a base64 PNG."""
    try:
        image_base64 = generate_awp_dashboard()
        return jsonify({'image': image_base64, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)