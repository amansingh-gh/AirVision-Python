import numpy as np
import requests
import joblib
import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 🔑 API KEY ---
# Make sure this key is active
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

        # 2. Current Pollution (Card Display)
        poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        poll_res = requests.get(poll_url).json()
        
        # --- 🟢 CHANGE START: Extract AQI ---
        # OpenWeatherMap provides AQI (1=Good, 5=Poor) inside ['list'][0]['main']['aqi']
        aqi_level = poll_res['list'][0]['main']['aqi'] 
        components = poll_res['list'][0]['components']
        
        current_data = {
            'aqi': aqi_level,  # <--- Added AQI here
            'pm2_5': components['pm2_5'], 'pm10': components['pm10'],
            'no2': components['no2'], 'co': components['co'],
            'so2': components['so2'], 'o3': components['o3'],
            'city_name': city_official_name
        }
        # --- 🟢 CHANGE END ---

        # 3. Current Weather (Card Display)
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        weather_res = requests.get(weather_url).json()
        
        current_data['h'] = weather_res['main']['humidity']
        current_data['w'] = weather_res['wind']['speed']
        current_data['temp'] = weather_res['main']['temp']

        # 4. AI FORECAST LOOP (Uses ONLY Weather)
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        forecast_res = requests.get(forecast_url).json()
        forecast_list = forecast_res['list'][:16] 

        predictions = []
        labels = []

        for item in forecast_list:
            # Timestamp (Label)
            time_str = item['dt_txt'].split(" ")[1][:5]
            
            # ---> REAL FUTURE WEATHER <---
            fut_temp = item['main']['temp']
            fut_humidity = item['main']['humidity']
            fut_wind = item['wind']['speed']
            
            # ---> FIX FOR "5 FEATURES ERROR" <---
            # Input format: [Temp, Humidity, Wind, Dummy_Target]
            features_for_scaling = np.array([[fut_temp, fut_humidity, fut_wind, 0]])
            
            if scaler and model:
                # 1. Scale (All 4 columns)
                features_scaled = scaler.transform(features_for_scaling)
                
                # 2. Extract only first 3 columns for Input (Ignore dummy target)
                input_scaled = features_scaled[:, 0:3]
                
                # 3. Reshape for LSTM (1 sample, 1 step, 3 features)
                lstm_in = input_scaled.reshape((1, 1, 3))
                
                # 4. Predict
                pred_scaled = model.predict(lstm_in)[0][0]
                
                # 5. Inverse Scale (To get real PM2.5 value back)
                pred_row = np.array([[0, 0, 0, pred_scaled]]) 
                pred_actual = scaler.inverse_transform(pred_row)[0][3]
                
                predictions.append(max(0, float(round(pred_actual, 2))))
            else:
                predictions.append(0)

            labels.append(time_str)

        return jsonify({
            'current': current_data,
            'forecast': { 'values': predictions, 'times': labels }
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)