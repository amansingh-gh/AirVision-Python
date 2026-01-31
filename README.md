# 🌍 AirVision Pro: AI-Powered Air Quality Monitor

**AirVision Pro** is a comprehensive web-based dashboard designed to monitor real-time Air Quality Index (AQI) and forecast pollution trends using Artificial Intelligence.

The system utilizes an **LSTM (Long Short-Term Memory)** Deep Learning model to predict PM2.5 levels for the upcoming 48 hours, helping users make informed health decisions.

<img width="1898" height="894" alt="image" src="https://github.com/user-attachments/assets/83bc14f7-dd11-4748-86b4-fcd34a397537" />
<img width="1907" height="911" alt="image" src="https://github.com/user-attachments/assets/2384a70e-747b-4da7-93f5-ae0db148405e" />



## ✨ Key Features

* **⚡ Real-Time Monitoring:** Live tracking of AQI, PM2.5, PM10, NO2, CO, Ozone, and weather conditions.
* **🤖 AI Forecasting:** Predicts air quality trends for the next **48 hours** using a trained LSTM model.
* **🎨 Glassmorphism UI:** A modern, responsive, and aesthetic dashboard interface.
* **🏥 Health Guide:** Dynamic health recommendations (e.g., "Wear Mask", "Ventilate Room") based on pollution severity.
* **📊 Interactive Charts:** Visualizes pollution spikes using **Chart.js** with hover interactions.
* **🌍 Global Coverage:** Fetches live data for any city using the **OpenWeatherMap API**.

---

## 🛠️ Tech Stack

### **Frontend**
* **HTML5, CSS3** (Glassmorphism Design)
* **JavaScript** (Fetch API, Dynamic DOM Manipulation)
* **Chart.js** (Data Visualization)
* **FontAwesome** (Icons)

### **Backend**
* **Python 3.x**
* **Flask** (Web Framework)
* **TensorFlow / Keras** (LSTM Model execution)
* **Numpy & Pandas** (Data Processing)
* **OpenWeatherMap API** (Live Data Source)

---

## 📂 Project Structure

```text
AirVision-Pro/
│
├── model/
│   ├── my_air_quality_lstm.h5   # Trained AI Model
│   └── scaler.pkl               # Data Scaler for Normalization
│
├── static/                      # CSS, Images, JS files
├── templates/
│   └── dashboard.html           # Main UI Dashboard
│
├── app.py                       # Flask Backend Application
├── requirements.txt             # Python Dependencies
└── README.md                    # Documentation
