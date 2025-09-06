# 🌍 World Bank Socio-Economic Data Analysis 📊

This project analyzes and models global socio-economic indicators using World Bank data, aiming to understand patterns, disparities, and to predict economic outcomes with machine learning techniques. Recent updates include configuration for deploying interactive dashboards with Streamlit.

## 📁 Dataset

- Sourced from the World Bank; multiple countries over different years.
- Indicators: GDP, GDP per capita, life expectancy, internet usage, birth/death rates, regions, and income groups.
- **Files**:
  - `WorldBank.xlsx`: Original dataset.
  - `WorldBank_Cleaned.csv`: Cleaned dataset after preprocessing.

---

## 🧹 Data Cleaning

- Dropped irrelevant columns (e.g., unemployment, population density).
- Forward/backward filled missing values by country/year.
- Removed rows with critical missing data.

---

## 🎯 Project Objectives

1. **GDP Trends by Income Group**: Visualized average GDP trends by income groups.
2. **Socio-Economic Indicators vs GDP**: Correlation heatmaps/scatter plots (internet usage, life expectancy vs GDP per capita).
3. **Regional Disparities**: Pie/bar charts for country count and regional averages (GDP per capita, internet usage, life expectancy).
4. **Income Group Disparities**: Bar plots comparing key indicators by income group.
5. **Socio-Economic Comparison by Region**: Bar plots for GDP per capita, internet usage, and life expectancy by region.

---

## 🤖 Machine Learning Models

- **Linear Regression**: Predicts GDP per capita using internet usage, life expectancy, birth rate. (R² ≈ 0.38)
- **Random Forest Regressor**: Improved accuracy, feature importance visualization.

---

## 🛠 Technologies & Configuration

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- Jupyter Notebook / VS Code
- Excel / CSV for data storage
- **Streamlit**: Interactive dashboard configuration initialized and themed.

---

## 📌 How to Run

1. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn openpyxl streamlit
   ```
2. Place `WorldBank.xlsx` in your working directory.
3. Run the analysis script or notebook:
   ```bash
   python world_bank_analysis.py
   ```
4. To launch the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

---

**Presentation Tip:**  
Present visuals objective by objective. Explain model logic and accuracy (R², MSE). Highlight insights such as how internet usage and life expectancy affect GDP per capita.
