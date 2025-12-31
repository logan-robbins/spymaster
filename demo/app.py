"""
TradingView Chart Demo with Forward Projections
Displays a 2-minute chart with forward projection lines based on average slope
"""
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

app = Flask(__name__)
CORS(app)

# Generate sample 2-minute OHLCV data
def generate_sample_data():
    """Generate realistic sample 2-minute bar data"""
    np.random.seed(42)
    start_time = datetime.now() - timedelta(hours=24)
    
    # Generate 720 bars (24 hours of 2-minute data)
    num_bars = 720
    timestamps = []
    bars = []
    
    base_price = 4500.0
    current_price = base_price
    
    for i in range(num_bars):
        bar_time = start_time + timedelta(minutes=2 * i)
        timestamp = int(bar_time.timestamp())
        
        # Random walk with drift
        change = np.random.normal(0, 2.0)
        current_price += change
        
        # Generate OHLC
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 1.5))
        low_price = open_price - abs(np.random.normal(0, 1.5))
        close_price = open_price + np.random.normal(0, 1.0)
        volume = np.random.uniform(100, 1000)
        
        current_price = close_price
        
        bars.append({
            'time': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })
    
    return bars

def generate_trading_day_data():
    """Generate a full trading day starting at 9:30 AM EST"""
    np.random.seed(42)
    
    # Create today's date at 9:30 AM EST
    from datetime import date
    import pytz
    
    est = pytz.timezone('America/New_York')
    today = date.today()
    market_open = est.localize(datetime.combine(today, datetime.strptime("09:30", "%H:%M").time()))
    
    # Generate 2-minute bars for a full trading day (9:30 AM - 4:00 PM = 6.5 hours = 195 bars)
    num_bars = 195
    bars = []
    
    base_price = 4500.0
    current_price = base_price
    
    for i in range(num_bars):
        bar_time = market_open + timedelta(minutes=2 * i)
        timestamp = int(bar_time.timestamp())
        
        # Add intraday patterns (higher volatility in morning/afternoon)
        hour = bar_time.hour
        volatility_factor = 1.5 if hour in [9, 10, 15] else 1.0
        
        # Random walk with drift
        change = np.random.normal(0, 2.0 * volatility_factor)
        current_price += change
        
        # Generate OHLC
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 1.5 * volatility_factor))
        low_price = open_price - abs(np.random.normal(0, 1.5 * volatility_factor))
        close_price = open_price + np.random.normal(0, 1.0 * volatility_factor)
        volume = np.random.uniform(100, 1000) * volatility_factor
        
        current_price = close_price
        
        bars.append({
            'time': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })
    
    return bars

# Store data in memory
SAMPLE_DATA = generate_sample_data()
TRADING_DAY_DATA = generate_trading_day_data()

@app.route('/')
def index():
    """Serve the main chart page"""
    return render_template('index.html')

@app.route('/config')
def config():
    """TradingView datafeed configuration"""
    return jsonify({
        'supported_resolutions': ['1', '2', '3', '5', '15', '30', '60', '240', 'D', 'W'],
        'supports_group_request': False,
        'supports_marks': False,
        'supports_search': True,
        'supports_timescale_marks': False,
    })

@app.route('/symbols')
def symbols():
    """Symbol info endpoint"""
    symbol = request.args.get('symbol', 'ES')
    
    return jsonify({
        'name': symbol,
        'ticker': symbol,
        'description': 'E-mini S&P 500 Futures',
        'type': 'futures',
        'session': '24x7',
        'exchange': 'CME',
        'listed_exchange': 'CME',
        'timezone': 'America/New_York',
        'format': 'price',
        'pricescale': 100,
        'minmov': 1,
        'has_intraday': True,
        'has_daily': True,
        'has_weekly_and_monthly': True,
        'supported_resolutions': ['1', '2', '3', '5', '15', '30', '60', '240', 'D', 'W'],
        'intraday_multipliers': ['1', '2', '3', '5', '15', '30', '60', '240'],
        'data_status': 'streaming',
    })

@app.route('/search')
def search():
    """Symbol search endpoint"""
    query = request.args.get('query', '')
    
    return jsonify([{
        'symbol': 'ES',
        'full_name': 'CME:ES',
        'description': 'E-mini S&P 500 Futures',
        'exchange': 'CME',
        'ticker': 'ES',
        'type': 'futures'
    }])

@app.route('/history')
def history():
    """Historical data endpoint"""
    symbol = request.args.get('symbol', 'ES')
    from_ts = int(request.args.get('from', 0))
    to_ts = int(request.args.get('to', time.time()))
    resolution = int(request.args.get('resolution', '2'))
    
    # Filter data based on time range
    filtered_bars = [
        bar for bar in SAMPLE_DATA
        if from_ts <= bar['time'] <= to_ts
    ]
    
    if not filtered_bars:
        return jsonify({
            's': 'no_data',
            'nextTime': None
        })
    
    # Aggregate to requested resolution if different from 2-min
    if resolution != 2:
        filtered_bars = aggregate_bars(filtered_bars, resolution)
    
    # Extract OHLCV arrays
    response = {
        's': 'ok',
        't': [bar['time'] for bar in filtered_bars],
        'o': [bar['open'] for bar in filtered_bars],
        'h': [bar['high'] for bar in filtered_bars],
        'l': [bar['low'] for bar in filtered_bars],
        'c': [bar['close'] for bar in filtered_bars],
        'v': [bar['volume'] for bar in filtered_bars],
    }
    
    return jsonify(response)

def aggregate_bars(bars, target_resolution_minutes):
    """Aggregate 2-minute bars into higher timeframes"""
    if not bars or target_resolution_minutes == 2:
        return bars
    
    aggregated = []
    bars_per_period = target_resolution_minutes // 2
    
    for i in range(0, len(bars), bars_per_period):
        period_bars = bars[i:i + bars_per_period]
        if not period_bars:
            continue
        
        # Aggregate OHLCV
        aggregated_bar = {
            'time': period_bars[0]['time'],
            'open': period_bars[0]['open'],
            'high': max(bar['high'] for bar in period_bars),
            'low': min(bar['low'] for bar in period_bars),
            'close': period_bars[-1]['close'],
            'volume': sum(bar['volume'] for bar in period_bars)
        }
        aggregated.append(aggregated_bar)
    
    return aggregated

@app.route('/replay/data')
def replay_data():
    """Get bars up to a specific index for replay mode"""
    bar_index = int(request.args.get('index', 0))
    
    # Return bars from start up to (and including) bar_index
    bars = TRADING_DAY_DATA[:bar_index + 1]
    
    if not bars:
        return jsonify({
            's': 'no_data',
            'total_bars': len(TRADING_DAY_DATA)
        })
    
    return jsonify({
        's': 'ok',
        't': [bar['time'] for bar in bars],
        'o': [bar['open'] for bar in bars],
        'h': [bar['high'] for bar in bars],
        'l': [bar['low'] for bar in bars],
        'c': [bar['close'] for bar in bars],
        'v': [bar['volume'] for bar in bars],
        'current_index': bar_index,
        'total_bars': len(TRADING_DAY_DATA)
    })

@app.route('/replay/projection')
def replay_projection():
    """Calculate projection bands for replay mode at a specific bar index"""
    bar_index = int(request.args.get('index', 10))
    lookback_bars = int(request.args.get('bars', 10))
    projection_bars = int(request.args.get('projection', 5))
    
    # Get all bars up to current index
    available_bars = TRADING_DAY_DATA[:bar_index + 1]
    
    # Use minimum 3 bars for lookback, or all available bars if less than requested
    min_lookback = 3
    if len(available_bars) < min_lookback:
        return jsonify({'error': f'Not enough data (need at least {min_lookback} bars)'})
    
    # Use requested lookback or all available bars (whichever is smaller)
    actual_lookback = min(lookback_bars, len(available_bars))
    recent_bars = available_bars[-actual_lookback:]
    
    # Calculate average slope using linear regression
    closes = np.array([bar['close'] for bar in recent_bars])
    time_indices = np.arange(len(closes))
    
    # Linear regression: y = mx + b
    slope, intercept = np.polyfit(time_indices, closes, 1)
    
    # Calculate standard deviation for confidence bands
    residuals = closes - (slope * time_indices + intercept)
    std_dev = np.std(residuals)
    band_width = 1.5 * std_dev
    
    # Last bar info
    last_bar = recent_bars[-1]
    last_time = last_bar['time']
    last_close = last_bar['close']
    
    # Calculate projection points for upper, middle, and lower bands
    resolution = 2  # 2-minute bars
    time_delta = resolution * 60
    upper_band = []
    middle_line = []
    lower_band = []
    
    # Include the starting point and project forward
    for i in range(projection_bars + 1):
        bar_index_proj = len(time_indices) - 1 + i
        bar_time = last_time + (i * time_delta)
        
        # Project the middle price based on slope
        middle_price = slope * bar_index_proj + intercept
        upper_price = middle_price + band_width
        lower_price = middle_price - band_width
        
        upper_band.append({'time': bar_time, 'value': round(upper_price, 2)})
        middle_line.append({'time': bar_time, 'value': round(middle_price, 2)})
        lower_band.append({'time': bar_time, 'value': round(lower_price, 2)})
    
    return jsonify({
        'bar_index': bar_index,
        'start_time': last_time,
        'start_price': round(last_close, 2),
        'end_time': upper_band[-1]['time'],
        'end_price': middle_line[-1]['value'],
        'slope': round(slope, 4),
        'band_width': round(band_width, 2),
        'upper_band': upper_band,
        'middle_line': middle_line,
        'lower_band': lower_band
    })

@app.route('/projection')
def projection():
    """Calculate projection bands with upper/lower bounds"""
    num_bars = int(request.args.get('bars', 10))
    projection_bars = int(request.args.get('projection', 5))
    resolution = int(request.args.get('resolution', 2))
    
    # Get aggregated data at the requested resolution
    all_bars = SAMPLE_DATA
    if resolution != 2:
        all_bars = aggregate_bars(all_bars, resolution)
    
    # Get last N bars
    recent_bars = all_bars[-num_bars:]
    
    if len(recent_bars) < 2:
        return jsonify({'error': 'Not enough data'})
    
    # Calculate average slope using linear regression
    times = np.array([bar['time'] for bar in recent_bars])
    closes = np.array([bar['close'] for bar in recent_bars])
    
    # Normalize times to bar indices for easier calculation
    time_indices = np.arange(len(times))
    
    # Linear regression: y = mx + b
    slope, intercept = np.polyfit(time_indices, closes, 1)
    
    # Calculate standard deviation for confidence bands
    residuals = closes - (slope * time_indices + intercept)
    std_dev = np.std(residuals)
    
    # Use 1.5x standard deviation for band width (roughly 87% confidence interval)
    band_width = 1.5 * std_dev
    
    # Last bar info
    last_bar = recent_bars[-1]
    last_time = last_bar['time']
    last_close = last_bar['close']
    
    # Calculate projection points for upper, middle, and lower bands
    time_delta = resolution * 60  # resolution in seconds
    upper_band = []
    middle_line = []
    lower_band = []
    
    # Include the starting point (last bar)
    for i in range(projection_bars + 1):  # Include start point
        bar_index = len(time_indices) - 1 + i
        bar_time = last_time + (i * time_delta)
        
        # Project the middle price based on slope
        middle_price = slope * bar_index + intercept
        
        # Calculate upper and lower bounds
        upper_price = middle_price + band_width
        lower_price = middle_price - band_width
        
        upper_band.append({
            'time': bar_time,
            'value': round(upper_price, 2)
        })
        
        middle_line.append({
            'time': bar_time,
            'value': round(middle_price, 2)
        })
        
        lower_band.append({
            'time': bar_time,
            'value': round(lower_price, 2)
        })
    
    return jsonify({
        'start_time': last_time,
        'start_price': round(last_close, 2),
        'end_time': upper_band[-1]['time'],
        'end_price': middle_line[-1]['value'],
        'slope': round(slope, 4),
        'band_width': round(band_width, 2),
        'std_dev': round(std_dev, 2),
        'bars_analyzed': num_bars,
        'projection_bars': projection_bars,
        'upper_band': upper_band,
        'middle_line': middle_line,
        'lower_band': lower_band
    })

if __name__ == '__main__':
    print("Starting TradingView Chart Demo...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)

