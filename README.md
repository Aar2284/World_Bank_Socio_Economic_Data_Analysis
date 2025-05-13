# 🌍 World Bank Socio-Economic Data Analysis 📊

This project involves the analysis and modeling of global socio-economic indicators using World Bank data. The objective is to understand patterns and disparities across countries, regions, and income groups, as well as to predict economic outcomes using machine learning techniques.

## 📁 Dataset

The dataset used in this project was sourced from the World Bank and includes data from multiple countries across different years. It contains indicators such as:

- GDP and GDP per capita (USD)
- Life expectancy at birth (years)
- Internet usage (% of population)
- Birth and death rates (per 1,000 people)
- Regional and income classifications

> **Note**: The original dataset file used is `WorldBank.xlsx`. A cleaned version is saved as `WorldBank_Cleaned.csv` after preprocessing.

---

## 🧹 Data Cleaning

- Dropped irrelevant columns (e.g., unemployment, population density).
- Forward and backward filled missing values grouped by country and year.
- Removed rows with critical missing data for analysis.

---

## 🎯 Project Objectives

### 📈 Objective 1: GDP Trends by Income Group

- Analyzed GDP and GDP per capita over time.
- Visualized average trends by income groups (Low, Lower-Middle, Upper-Middle, High).

### 📊 Objective 2: Socio-Economic Indicators vs GDP

- Correlation heatmap and scatter plots to show the relationship between:
  - Internet usage & GDP per capita
  - Life expectancy & GDP per capita

### 🌍 Objective 3: Regional Disparities

- Pie and bar charts displaying:
  - Number of countries per region
  - Regional averages of GDP per capita, internet usage, and life expectancy

### 💵 Objective 4: Income Group Disparities

- Compared average values of GDP per capita, internet usage, and life expectancy across different income groups using bar plots.

### 🌐 Objective 5: Socio-Economic Comparison by Region

- Bar plots to compare GDP per capita, internet usage, and life expectancy by region.

---

## 🤖 Objective 6: Machine Learning Models

### 🔢 Linear Regression

- **Goal**: Predict `GDP per capita (USD)`
- **Features**: 
  - Internet usage (%)
  - Life expectancy (years)
  - Birth rate
- **Evaluation**:
  - R² score: ~0.38 (moderate prediction capability)
  - Mean Squared Error (MSE)

### 🌲 Random Forest Regressor

- Improved performance using ensemble learning.
- Displays feature importance.
- Higher accuracy compared to linear regression.

---

## 🛠 Technologies Used

- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Jupyter Notebook / VS Code**
- **Excel / CSV** for data storage and exploration

---

## 📌 How to Run

1. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn openpyxl

Place WorldBank.xlsx in your working directory.
Run the Python script or Jupyter notebook:
  -python world_bank_analysis.py

**Presentation Tip**
Present visuals objective by objective.
Explain model logic and accuracy (R², MSE).
Highlight insights (e.g., how internet usage and life expectancy affect GDP per capita).

