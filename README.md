# 📈 Mastercard Stock Price Prediction (Work in Progress → Deployed Prototype)

This repository contains my end-to-end journey of building a **stock price movement prediction model for Mastercard (MA)**, starting from raw historical data exploration → machine learning modeling → evaluation → deployment with **Streamlit**.

The goal is to predict whether the stock will go **UP or DOWN the next day**, and provide **probability estimates** for informed decision-making.

---

## 🔹 Project Workflow  

### 1. Data Collection & Cleaning  
- Imported historical Mastercard stock data (Open, High, Low, Close, Volume, Dividends).  
- Generated lag features (`Close_Lag1`, `Return_Lag1`, `Return_Lag2`, …).  
- Added technical indicators:  
  - Moving Averages (**MA7, MA30**)  
  - Volatility measures (**Volatility_7**)  
  - Trend & ratio features (**Trend_2, Close_Ratio_5, Trend_60, …**)  
- Cleaned null values and standardized numeric features.  

### 2. Exploratory Data Analysis (EDA)  
- 📊 Visualized stock price trends, moving averages, and rolling volatility.  
- 📌 Distribution analysis of returns.  
- 🔗 Correlation heatmaps to identify key relationships.  

### 3. Baseline Modeling  
- ✅ **Linear Regression** → tested continuous price prediction.  
- ✅ **Random Forest Regressor** → slightly better, but unstable.  
- 📏 Metrics: **R², MAE** (~57% classification equivalent, not reliable).  

### 4. Advanced Modeling with LightGBM  
- Reformulated as a **binary classification problem** (1 = UP, 0 = DOWN).  
- Tuned hyperparameters: `learning_rate`, `num_leaves`, `n_estimators`.  
- Used **Stratified K-Fold Cross Validation** for evaluation.  

**Final Metrics:**  
- Accuracy: **~68%**  
- ROC-AUC: **~0.71**  
- Balanced precision & recall.  

### 5. Challenges & Learnings  
- ❌ **Feature explosion** → solved with feature selection & importance ranking.  
- ❌ **Overfitting** → handled via early stopping & regularization.  
- ❌ **Inconsistent performance** → stabilized with cross-validation.  
- 💡 **Key takeaway:** Feature engineering improved performance more than model complexity.  

### 6. Deployment with Streamlit  
Developed an interactive web app with two modes:  

- 📂 **Batch Prediction Mode** – Upload CSV → get predictions with probabilities.  
- 📝 **Single-Row Input Mode** – Enter stock data manually → instant prediction.  

**Example Output:**  
Prediction: 0 (Down)
Probability (Up): 0.467

**App Features:**  
- Data upload interface  
- Batch predictions with probability outputs  
- Single-row manual input form  
- Example prediction results  

---

## 🔹 Tech Stack  
- **Language & Environment:** Python (Jupyter Notebook, Streamlit)  
- **Libraries:**  
  - Data Processing → Pandas, NumPy  
  - Visualization → Matplotlib, Seaborn  
  - Modeling → Scikit-learn, LightGBM, Joblib  
  - Deployment → Streamlit  

---

## 🔹 Next Steps  
- ⏳ Deploy app on **Streamlit Cloud** for public access  
- ⏳ Add visual probability trend graphs for predictions  
- ⏳ Experiment with **LSTM/GRU** for time-series forecasting  
- ⏳ Improve feature set with advanced indicators (**RSI, MACD, Bollinger Bands**)  

---

## 📌 How to Run Locally  

Clone the repo:  
```bash
git clone https://github.com/<your-username>/mastercard-stock-prediction.git
cd mastercard-stock-prediction

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run streamlit_app.py

Enter inputs or upload CSV to test predictions.

📖 Project Status

✔ Data Processing & Feature Engineering Completed
✔ Model Training with LightGBM Completed
✔ Streamlit Deployment Prototype Ready
🚀 Next Goal → Public Deployment

📌 This project reflects my learning journey in applied ML & deployment, showing how I improved step by step from a basic regression baseline to an interactive prediction tool.
