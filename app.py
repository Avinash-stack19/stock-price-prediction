from flask import Flask, render_template, request, make_response, url_for
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
import pmdarima as pm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input  # Add this with other imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import pdfkit
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input  # Add this with other imports
import logging


app = Flask(__name__)

# ✅ Fixed Windows path with raw string
PDF_CONFIG = pdfkit.configuration(wkhtmltopdf=r'E:\coding\stock-price-prediction\wkhtmltox\bin\wkhtmltopdf.exe')

def number_format(value, decimals=2):
    """Jinja2 filter for number formatting"""
    try:
        return f"{float(value):,.{decimals}f}"
    except:
        return value

app.jinja_env.filters['number_format'] = number_format

# --------------------------
# 🛠 Enhanced Utility Functions
# --------------------------
def create_features(df):
    """Create advanced technical indicators"""
    df = df.copy()
    # Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # Momentum Indicators
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['RSI'] = compute_rsi(df['Close'], window=14)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df['Bollinger_Upper'] = df['MA_20'] + 2*df['Volatility']
    df['Bollinger_Lower'] = df['MA_20'] - 2*df['Volatility']
    
    # Trend Indicators
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_metrics(y_true, y_pred):
    """Enhanced metrics calculation with accuracy percentage and safe division"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0, 'accuracy_pct': 0}
    
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    
    # Directional accuracy (up/down match)
    correct_directions = 0
    if len(y_true) > 1:
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        correct_directions = np.mean(true_directions == pred_directions) * 100

    # Value proximity accuracy (within 5%)
    threshold = 0.05  # 5% threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error = np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))
    
    valid_errors = relative_error[np.isfinite(relative_error)]
    proximity_accuracy = np.mean(valid_errors <= threshold) * 100 if len(valid_errors) > 0 else 0

    # Combined weighted accuracy
    final_accuracy = (0.6 * correct_directions) + (0.4 * proximity_accuracy)

    return {
        'mae': round(mean_absolute_error(y_true, y_pred), 4),
        'mse': round(mean_squared_error(y_true, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'r2': round(r2_score(y_true, y_pred), 4),
        'accuracy_pct': round(final_accuracy, 2)
    }


def process_stock_data(selected_stock):
    """Process data with enhanced features and model selection"""
    if not selected_stock.endswith('.csv'):
        selected_stock += '.csv'
        
    df = pd.read_csv(os.path.join("data", selected_stock))
    
    # Enhanced data preparation
    df['Date'] = pd.to_datetime(df['Date'])
    df = create_features(df)
    df = df.iloc[-365*2:]  # Use 2 years of data for better trend analysis
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    today_data = df.iloc[-1].to_dict()
    numeric_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    for key in numeric_fields:
        today_data[key] = float(today_data.get(key, 0))
    
    # Model Predictions with enhanced configurations
    arima_forecast, arima_metrics = arima_predict(df)
    lstm_forecast, lstm_metrics = lstm_predict(df, selected_stock.replace('.csv', ''))
    lr_forecast, lr_metrics = linear_regression_predict(df)
    rf_forecast, rf_metrics = random_forest_predict(df)

    # Best Model Selection based on accuracy
    models = {
        'ARIMA': arima_metrics,
        'LSTM': lstm_metrics,
        'Linear Regression': lr_metrics,
        'Random Forest': rf_metrics
    }
    
    best_model_name = max(models, key=lambda x: models[x]['accuracy_pct'])
    best_forecast = {
        'ARIMA': arima_forecast,
        'LSTM': lstm_forecast,
        'Linear Regression': lr_forecast,
        'Random Forest': rf_forecast
    }[best_model_name]
    
    # Recommendation Logic with confidence check
    current_close = today_data['Close']
    next_day_preds = [pred[0]['Predicted_Close'] for pred in [arima_forecast, lstm_forecast, lr_forecast, rf_forecast]]
    avg_pred = np.mean(next_day_preds)
    
    # Confidence threshold (2% difference)
    confidence_threshold = 0.02 * current_close
    recommendation = "HOLD"
    if abs(avg_pred - current_close) > confidence_threshold:
        recommendation = "BUY" if avg_pred > current_close else "SELL"
    
    return {
        'stock_name': selected_stock.replace('.csv', ''),
        'today_data': today_data,
        'arima_metrics': arima_metrics,
        'lstm_metrics': lstm_metrics,
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'arima_forecast': arima_forecast,
        'lstm_forecast': lstm_forecast,
        'lr_forecast': lr_forecast,
        'rf_forecast': rf_forecast,
        'best_model_name': best_model_name,
        'best_accuracy': models[best_model_name]['accuracy_pct'],
        'best_forecast': best_forecast,
        'recommendation': recommendation
    }

# --------------------------
# 🚀 Flask Routes (No changes needed here)
# --------------------------

@app.route('/')
def index():
    stock_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    return render_template('index.html', csv_files=stock_files)

@app.route('/predict', methods=['POST'])
def predict():
    selected_stock = request.form['company']
    try:
        data = process_stock_data(selected_stock)
        return render_template(
            'result.html',
            plot_paths={
                'arima': 'arima_plot.png',
                'lstm': 'lstm_plot.png',
                'linear_regression': 'linear_regression_plot.png',
                'random_forest': 'random_forest_plot.png'
            },
            **data
        )
    except Exception as e:
        return f"Error processing request: {str(e)}", 500


@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        selected_stock = request.form.get('selected_stock')
        if not selected_stock:
            return "Missing stock parameter", 400

        if not selected_stock.endswith('.csv'):
            selected_stock += '.csv'
            
            
        
        # Process data
        data = process_stock_data(selected_stock)
        print("✅ Today Data Sent to PDF:", data['today_data'])
        # Render HTML for PDF
        rendered = render_template(
            'pdf_template.html',
            plot_paths={
                'arima': os.path.abspath("static/arima_plot.png"),
                'lstm': os.path.abspath("static/lstm_plot.png"),
                'linear_regression': os.path.abspath("static/linear_regression_plot.png"),
                'random_forest': os.path.abspath("static/random_forest_plot.png")
            },
            **data
        )

        # Generate PDF with proper config and options
        pdf = pdfkit.from_string(rendered, False, configuration=PDF_CONFIG, options={
            'enable-local-file-access': '',
            'page-size': 'A4',
            'encoding': 'UTF-8'
        })

        # Send as downloadable PDF
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename={data["stock_name"]}_report.pdf'
        return response

    except Exception as e:
        import traceback
        print("❌ PDF Generation Error:", str(e))
        print(traceback.format_exc())
        return f"Error generating PDF: {str(e)}", 500


# --------------------------
# 🤖 Enhanced Model Implementations
# --------------------------
def arima_predict(df):
    try:
        # Enhanced ARIMA configuration
        model = pm.auto_arima(
            df['Close'],
            seasonal=True,
            m=7,
            suppress_warnings=True,
            stepwise=True,
            error_action='ignore',
            trace=True,
            with_intercept=True
        )
        
        # Generate forecast with confidence intervals
        forecast, conf_int = model.predict(
            n_periods=7,
            return_conf_int=True,
            alpha=0.05  # 95% confidence interval
        )
        
        # Plotting
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'], df['Close'], label='Historical')
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1,8)]
        plt.plot(future_dates, forecast, label='ARIMA Forecast')
        plt.fill_between(future_dates, conf_int[:,0], conf_int[:,1], alpha=0.1)
        plt.title(f'ARIMA Forecast ({model.order})')
        plt.legend()
        plt.savefig('static/arima_plot.png')
        plt.close()
        
        return [
            {'Date': d.strftime('%Y-%m-%d'), 'Predicted_Close': float(v)}
            for d, v in zip(future_dates, forecast)
        ], calculate_metrics(df['Close'][-len(forecast):].values, forecast)
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.error("ARIMA model failed", exc_info=True)

    # Optional: write fallback data to a file or notify frontend
    return [
        {'Date': (df['Date'].iloc[-1] + timedelta(days=i)).strftime('%Y-%m-%d'), 'Predicted_Close': None}
        for i in range(1, 8)
    ], {
        'mae': -1,
        'mse': -1,
        'rmse': -1,
        'r2': -1,
        'accuracy_pct': 0
    }


def lstm_predict(df, stock_name):
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)

        # Prepare model save path
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"lstm_{stock_name}.h5")

        lookback = 60
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])

        # Create training data
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Load or train model
        if os.path.exists(model_path):
            logging.info(f"Loading cached LSTM model: {model_path}")
            model = load_model(model_path)
        else:
            logging.info(f"Training new LSTM model: {model_path}")
            model = Sequential([
               Input(shape=(X.shape[1], 1)),  # Add this line
               LSTM(128, return_sequences=True),
               Dropout(0.4),
               LSTM(64, return_sequences=False),
               Dropout(0.3),
               Dense(25),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=10)

            model.fit(
                X, y,
                batch_size=32,
                epochs=100,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

            model.save(model_path)
            logging.info(f"LSTM model saved to {model_path}")

        # Forecast next 7 days
        inputs = scaled_data[-lookback:]
        forecasts = []
        for _ in range(7):
            x = inputs[-lookback:].reshape(1, lookback, 1)
            pred = model.predict(x, verbose=0)[0][0]
            forecasts.append(pred)
            inputs = np.append(inputs, pred)

        forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 8)]

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Historical')
        plt.plot(future_dates, forecasts, label='LSTM Forecast')
        plt.title('Enhanced LSTM Predictions')
        plt.legend()
        plt.savefig('static/lstm_plot.png')
        plt.close()

        return [
            {'Date': d.strftime('%Y-%m-%d'), 'Predicted_Close': float(v[0])}
            for d, v in zip(future_dates, forecasts)
        ], calculate_metrics(df['Close'][-len(forecasts):].values, forecasts.flatten())

    except Exception as e:
        logging.error("LSTM model failed", exc_info=True)
        return [
            {'Date': (df['Date'].iloc[-1] + timedelta(days=i)).strftime('%Y-%m-%d'), 'Predicted_Close': None}
            for i in range(1, 8)
        ], {
            'mae': -1,
            'mse': -1,
            'rmse': -1,
            'r2': -1,
            'accuracy_pct': 0
        }
def linear_regression_predict(df):
    try:
        # Enhanced feature set
        X = df[['MA_7', 'MA_30', 'Momentum', 'Volatility', 'RSI', 'MACD']]
        y = df['Close']
        
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics.append(calculate_metrics(y_test.values, preds))
        
        # Final model with all data
        model = LinearRegression()
        model.fit(X, y)
        forecast = model.predict(X[-7:])
        
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1,8)]
        
        # Plotting
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'], df['Close'], label='Historical')
        plt.plot(future_dates, forecast, label='Linear Regression Forecast')
        plt.title('Enhanced Linear Regression Predictions')
        plt.legend()
        plt.savefig('static/linear_regression_plot.png')
        plt.close()
        
        return [
            {'Date': d.strftime('%Y-%m-%d'), 'Predicted_Close': float(v)}
            for d, v in zip(future_dates, forecast)
        ], {
            'mae': np.mean([m['mae'] for m in metrics]),
            'mse': np.mean([m['mse'] for m in metrics]),
            'rmse': np.mean([m['rmse'] for m in metrics]),
            'r2': np.mean([m['r2'] for m in metrics]),
            'accuracy_pct': np.mean([m['accuracy_pct'] for m in metrics])
        }
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.error("Linear Regression model failed", exc_info=True)

    return [
        {'Date': (df['Date'].iloc[-1] + timedelta(days=i)).strftime('%Y-%m-%d'), 'Predicted_Close': None}
        for i in range(1, 8)
    ], {
        'mae': -1,
        'mse': -1,
        'rmse': -1,
        'r2': -1,
        'accuracy_pct': 0
    }
def random_forest_predict(df):
    try:
        # Enhanced feature set
        X = df[['MA_7', 'MA_30', 'Momentum', 'Volatility', 'RSI', 'MACD', 'Volume']]
        y = df['Close']
        
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics.append(calculate_metrics(y_test.values, preds))
        
        # Final model
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X, y)
        forecast = model.predict(X[-7:])
        
        future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1,8)]
        
        # Plotting
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'], df['Close'], label='Historical')
        plt.plot(future_dates, forecast, label='Random Forest Forecast')
        plt.title('Enhanced Random Forest Predictions')
        plt.legend()
        plt.savefig('static/random_forest_plot.png')
        plt.close()
        
        return [
            {'Date': d.strftime('%Y-%m-%d'), 'Predicted_Close': float(v)}
            for d, v in zip(future_dates, forecast)
        ], {
            'mae': np.mean([m['mae'] for m in metrics]),
            'mse': np.mean([m['mse'] for m in metrics]),
            'rmse': np.mean([m['rmse'] for m in metrics]),
            'r2': np.mean([m['r2'] for m in metrics]),
            'accuracy_pct': np.mean([m['accuracy_pct'] for m in metrics])
        }
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.error("Random Forest model failed", exc_info=True)

    return [
        {'Date': (df['Date'].iloc[-1] + timedelta(days=i)).strftime('%Y-%m-%d'), 'Predicted_Close': None}
        for i in range(1, 8)
    ], {
        'mae': -1,
        'mse': -1,
        'rmse': -1,
        'r2': -1,
        'accuracy_pct': 0
    }
if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, threaded=False)