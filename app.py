import numpy as np
import requests
import joblib
import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 🔑 API KEY ---
API_KEY = "491b1a6eec1c098ee85934c9c48aaea9"  

# --- AI CONFIG ---
MODEL_PATH = 'model/my_air_quality_lstm.h5'
SCALER_PATH = 'model/scaler.pkl'

model = None
scaler = None

def load_ai_assets():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("✅ AI System Loaded (Inputs: Temp, Humidity, Wind).")
        else:
            print("⚠️ AI Files missing.")
    except Exception as e:
        print(f"❌ Error loading AI: {e}")

load_ai_assets()

# --- ROUTES ---

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    city = request.args.get('city') 
    return render_template('dashboard.html', city=city)

@app.route('/api/get_full_data', methods=['POST'])
def get_full_data():
    city = request.form.get('city')
    if not city: return jsonify({'error': 'City required'}), 400

    try:
        # 1. Geocoding
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_res = requests.get(geo_url).json()
        if not geo_res: return jsonify({'error': f"City '{city}' not found."}), 404
        
        lat = geo_res[0]['lat']
        lon = geo_res[0]['lon']
        city_official_name = geo_res[0]['name']

        # 2. Current Pollution
        poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        poll_res = requests.get(poll_url).json()
        
        aqi_level = poll_res['list'][0]['main']['aqi'] 
        components = poll_res['list'][0]['components']
        current_pm25 = components['pm2_5'] # LIVE PM2.5 DATA
        
        current_data = {
            'aqi': aqi_level, 
            'pm2_5': current_pm25, 'pm10': components['pm10'],
            'no2': components['no2'], 'co': components['co'],
            'so2': components['so2'], 'o3': components['o3'],
            'city_name': city_official_name
        }

        # 3. Current Weather
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        weather_res = requests.get(weather_url).json()
        
        current_data['h'] = weather_res['main']['humidity']
        current_data['w'] = weather_res['wind']['speed']
        current_data['temp'] = weather_res['main']['temp']

        # 4. AI FORECAST (Uses only 8 steps for 24 Hours)
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        forecast_res = requests.get(forecast_url).json()
        forecast_list = forecast_res['list'][:8] 

        raw_predictions = []
        labels = []

        for item in forecast_list:
            time_str = item['dt_txt'].split(" ")[1][:5]
            fut_temp = item['main']['temp']
            fut_humidity = item['main']['humidity']
            fut_wind = item['wind']['speed']
            
            features_for_scaling = np.array([[fut_temp, fut_humidity, fut_wind, 0]])
            
            if scaler and model:
                features_scaled = scaler.transform(features_for_scaling)
                input_scaled = features_scaled[:, 0:3]
                lstm_in = input_scaled.reshape((1, 1, 3))
                pred_scaled = model.predict(lstm_in)[0][0]
                pred_row = np.array([[0, 0, 0, pred_scaled]]) 
                pred_actual = scaler.inverse_transform(pred_row)[0][3]
                
                # Minimum 1.0 to avoid division by zero later
                raw_predictions.append(max(1.0, float(round(pred_actual, 2))))
            else:
                raw_predictions.append(1.0)

            labels.append(time_str)

        # --- 🛠️ ML FIX: BASELINE CALIBRATION ---
        # Scale the model's predictions to match the city's ACTUAL baseline
        final_predictions = []
        if len(raw_predictions) > 0:
            first_raw_pred = raw_predictions[0]
            # Calculate Ratio (e.g., Oslo current is 8, Model says 200 -> Ratio is 0.04)
            calibration_ratio = current_pm25 / first_raw_pred if first_raw_pred > 0 else 1
            
            # Apply ratio to all future predictions
            for p in raw_predictions:
                calibrated_value = round(p * calibration_ratio, 2)
                final_predictions.append(calibrated_value)
        else:
            final_predictions = raw_predictions

        return jsonify({
            'current': current_data,
            'forecast': { 'values': final_predictions, 'times': labels }
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)