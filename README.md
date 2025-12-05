# SHEOS - Solar Energy Optimiser System

**SHEOS** is an intelligent, AI-driven solar energy management platform designed to help homeowners maximize their solar investment. Built with Python and Streamlit, it uses Machine Learning to predict solar generation, optimize appliance schedules, and track financial ROI in real-time.

---

## 🚀 Features

### 1. 🏠 Interactive Dashboard
- **Real-time Monitoring:** View live power generation, system efficiency, and grid pricing.
- **Visual Analytics:** Gauge charts and speedometers for system health and current output.
- **Environmental Impact:** Track CO₂ emissions saved and equivalent trees planted.

### 2. 📈 AI Solar Forecasting
- **ML-Powered Predictions:** Uses a **Random Forest Regressor** trained on historical weather data (Irradiance, Temperature, Cloud Cover) to forecast generation.
- **24-Hour Timeline:** Visual graphs showing expected power output vs. weather conditions for the day ahead.

### 3. ⚡ Smart Load Monitor
- **Appliance Calculator:** Input usage for ACs, Fans, LEDs, and Washing Machines to calculate total household load.
- **Net Metering:** Instantly see if you are running on **Solar Surplus** (Free) or **Grid Import** (Costly).
- **Self-Sufficiency Gauge:** Visualizes what percentage of your energy needs are met by solar.

### 4. 🧠 Intelligent Scheduler
- **Optimization Engine:** Analyzes upcoming weather patterns to recommend the **"Best Time Slots"** for high-energy tasks (e.g., EV charging, laundry).
- **Heatmap Visualization:** Color-coded timeline identifying peak generation hours.

### 5. 💰 ROI & Financial Analysis
- **Cost Simulation:** Runs a 30-day simulation comparing "Grid Only" vs. "Solar + AI" costs.
- **Investment Calculator:** Estimates payback period and 5-year profit based on panel installation costs.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (Ultra Dark Neon Theme)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest Regressor)
- **Visualization:** Plotly Express, Plotly Graph Objects

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Omakshat2005/Solar-Energy-Analyser.git](https://github.com/Omakshat2005/Solar-Energy-Analyser.git)
   cd Solar-Energy-Analyser
