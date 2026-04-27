# 🍭 Nassau Candy: Factory Reallocation & Shipping Optimization System

### 🚀 Project Overview
This project is a **Prescriptive Decision Intelligence System** built during my Data Science internship at Unified Mentor. It transitions Nassau Candy Distributor from legacy, rule-based shipping to an AI-driven optimization framework.

The system uses a **Random Forest Regressor** to predict shipping lead times and a **multi-objective optimization algorithm** to recommend the best factory for specific product-region pairs, balancing speed and profitability.

---

### 🛠️ Key Features
- **Factory Optimization Simulator:** Compare current vs. predicted performance across 5 national factories.
- **What-If Scenario Analysis:** Interactive geospatial mapping and distance-correlation analysis.
- **Strategic Weighting Slider:** Granular control to toggle between "Speed Focused" and "Profit Focused" strategies.
- **Risk & Impact Panel:** Automated alerts for margin erosion and high-risk logistical reassignments.
- **Performance Metrics:** Real-time model reliability tracking (MAE, RMSE, R²).

---

### 📊 Model Performance
- **Algorithm:** Random Forest Regressor
- **Mean Absolute Error (MAE):** 0.88 Days
- **RMSE:** 0.92 Days
- **R² Score:** 0.59
- **Impact:** Identified a potential **18.4% reduction** in lead times for critical routes.

---

### 💻 Tech Stack
- **Language:** Python 3.12
- **AI/ML:** Scikit-learn, Joblib
- **Data:** Pandas, NumPy
- **Visuals:** Plotly Express, Seaborn, Matplotlib
- **Deployment:** Streamlit

---

### 📦 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Vipsa2207/ Nassau-Candy-Supply-Chain-Optimizer.git
