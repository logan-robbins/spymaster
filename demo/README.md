# TradingView Forward Projection Demo - Replay Mode

This demo showcases a chart with forward projection bands that accumulate throughout a trading day replay.

## Features

### Replay Mode (Default)
- **Trading Day Replay**: Watch a full 6.5-hour trading day (9:30 AM - 4:00 PM EST) unfold at 16x speed
- **Progressive Bar Display**: 195 two-minute bars appear every 7.5 seconds
- **2-Overlay System**: Clean visualization with only two projection overlays:
  - **Forward Projection** (bright orange/peach): Current prediction extending into future
  - **Historical Accumulation** (faded orange/peach): Continuous band showing all past predictions
- **Early Projections**: First projection appears after just 3 bars (minimum lookback requirement)
- **Auto-Updating Projections**: Forward projection updates automatically at each bar close for next P bars
- **Smooth Historical Band**: As each projection completes, it extends the historical band smoothly (no jagged overlaps)
- **Future Time Axis**: Chart shows at least 1 hour of timestamps into the future
- **Replay Controls**: Start, Pause, and Reset buttons for full control
- **Complete in ~24 minutes**: Watch an entire trading day in under half an hour

### Static Mode (Legacy)
- **Multi-Timeframe Candlestick Chart**: Displays ES futures data with configurable timeframes
- **Single Projection Band**: Shows current projection with orange lines and translucent peach fill
- **Manual Updates**: Update projections on demand

### Core Features (Both Modes)
- **Projection Bands**: Upper and lower confidence bounds with orange lines and translucent peach fill
- **Statistical Confidence**: Uses standard deviation to calculate band width (±1.5σ)
- **Average Slope Calculation**: Linear regression on the last N bars (default: 10)
- **Configurable Settings**: Adjust lookback period (2-100 bars) and projection length (1-20 bars)
- **American Time Format**: 12-hour clock format with AM/PM on axis labels
- **Real-time Statistics**: Shows progress, slope, band width, current price, and latest projection

## How It Works

### Replay Mode Workflow

1. **Trading Day Generation**: Creates 195 two-minute bars spanning 9:30 AM - 4:00 PM EST with realistic intraday volatility patterns

2. **Progressive Playback**: Bars appear sequentially at 16x speed (7.5 seconds per bar)

3. **2-Overlay System**: Uses only two projection overlays for clean visualization:
   - **Forward Projection** (bright): Shows current prediction into future
   - **Historical Band** (faded): Accumulates all past predictions into one continuous band

4. **Auto-Updating Projections**: At each bar close (starting from bar 3):
   - Analyzes last N bar closes using linear regression (minimum 3 bars)
   - Calculates ±1.5σ confidence bands from the trend line
   - **Extends Historical**: Takes the starting point of the old projection and adds it to the historical accumulator
   - **Updates Forward**: Replaces forward projection with new P-bar forecast
   - Result: Smooth, continuous historical band (no jagged overlaps)

5. **Visualization**:
   - Forward projection is bright orange/peach (shows current expectation)
   - Historical band is faded orange/peach (shows evolution of predictions)
   - Time axis extends 1 hour into future to show full forward projection
   - Only 2 overlays total = clean, readable chart

6. **Result**: By market close, one smooth historical band traces the complete evolution of predictions throughout the day

### Visualization Layers

- **Candlestick Series**: Real OHLCV bars that appear progressively
- **Forward Projection** (bright overlay):
  - Bright orange boundary lines (upper and lower)
  - Translucent peach fill (shows confidence region)
  - Extends into future (visible on time axis)
- **Historical Band** (faded overlay):
  - Faded orange boundary lines
  - Light peach fill
  - Continuous accumulation of all past predictions
  - Shows smooth evolution of market expectations
- **Time Axis**: Shows at least 1 hour into future to display full projection
- **Real-time Stats**: Progress, current slope, band width, and forward projection price

## Setup & Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser to: http://localhost:5000

## Using Replay Mode

1. Open the demo in your browser
2. Ensure "Replay" mode is selected (default)
3. Adjust **Lookback Bars** (how many bars to analyze) and **Projection Bars** (how far to project)
4. Click **▶ Start Replay (16x)** to begin
5. Watch as:
   - Bars appear every 7.5 seconds at 16x speed
   - First projection appears after just 3 bars
   - New projection bands automatically created at each subsequent bar close
   - All historical projection bands remain visible with fading opacity
   - Progress indicator shows current position (e.g., "75/195 (38.5%)")
6. Click **⏸ Pause** to pause playback (resume with Start)
7. Click **↻ Reset** to clear all data and start over
8. Switch to **Static** mode for traditional manual projection updates

**Full replay completes in approximately 24 minutes** (195 bars × 7.5 seconds)

## What You'll See

### Replay Visualization

As the replay progresses, you'll observe:

1. **Growing Chart**: Candlesticks appear left-to-right at 16x speed, rapidly building the trading day
2. **Future Time Axis**: Chart automatically shows 1+ hour of timestamps into the future
3. **Early Projections**: First projection appears after just 3 bars (9:36 AM)
4. **Two Clean Overlays**:
   - **Bright Forward Projection**: Updates at each bar close, extending P bars into future
   - **Faded Historical Band**: Grows smoothly as old projections are added to it
5. **Smooth Accumulation**: Historical band extends continuously (no jagged overlaps or clutter)
6. **Clear Visualization**: Only 2 overlays means the chart stays readable throughout the day
7. **Pattern Recognition**: By end of day, the historical band shows:
   - Where predictions were consistently accurate (narrow band following price)
   - Where predictions were volatile (wider band)
   - How confidence changed throughout the day (band width variations)
   - Trend changes (band direction shifts)

### Key Visual Elements

- **Green/Red Candlesticks**: Actual price bars (green = up, red = down)
- **Bright Orange Lines**: Upper and lower bounds of forward projection
- **Bright Peach Fill**: Confidence region of forward projection
- **Faded Orange Lines**: Upper and lower bounds of historical band
- **Faded Peach Fill**: Historical confidence region showing past predictions
- **Time Axis**: 12-hour format, extends 1+ hour into future to show projections
- **Only 2 Overlays**: Clean visualization without clutter

## API Endpoints

### Replay Mode Endpoints
- `/replay/data` - Get bars up to specified index for progressive playback
- `/replay/projection` - Calculate projection bands at specific bar index

### Static Mode Endpoints
- `/` - Main chart interface
- `/config` - TradingView datafeed configuration
- `/symbols` - Symbol information
- `/history` - Historical OHLCV data
- `/projection` - Calculate forward projection based on slope

## Configuration

### Replay Mode Settings
- **Lookback Bars**: Number of historical bars to analyze for each projection (default: 10, range: 2-100)
  - Projections start after minimum 3 bars (uses all available data if less than requested lookback)
  - Auto-updates at every bar close once minimum requirement is met
- **Projection Bars**: Number of bars to project forward from each point (default: 5, range: 1-20)
- **Playback Speed**: Fixed at 16x (7.5 seconds per bar, ~24 minutes total)
- **Mode Switch**: Toggle between Replay and Static modes

### Static Mode Settings
- **Resolution**: Choose timeframe (2min, 5min, 15min, 30min, 1hr, 4hr, 8hr)
- **Lookback Bars**: Same as replay mode
- **Projection Bars**: Same as replay mode
- **Manual Update**: Click to refresh single projection band

## Technical Details

The projection uses linear regression with confidence bands:
- **Trend Line**: `y = mx + b` where m is the slope
- **Confidence Bands**: `upper/lower = trend ± (1.5 × std_deviation)`
- **Standard Deviation**: Calculated from residuals of the linear fit
- **Minimum Lookback**: 3 bars (starts projections early)
- **Adaptive Lookback**: Uses requested lookback or all available bars (whichever is smaller)
- **Auto-Update**: Projection regenerated at every bar close for next P bars
- **Time Projection**: `future_time = last_bar_time + (i × resolution × 60 seconds)`
- **Playback Speed**: 16x (2-minute bar in 7.5 real-time seconds)
- **Visualization**: Layered area series create the filled band effect, with line series for boundaries

## Chart Library

Uses [TradingView Lightweight Charts](https://www.tradingview.com/lightweight-charts/) for rendering, which provides:
- Professional candlestick visualization
- Smooth zooming and panning
- Customizable line styles (dotted for projections)
- Dark theme matching TradingView aesthetic

