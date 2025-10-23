# 🌍 World Bank Socio-Economic Data Analysis 📊

[![Stars](https://img.shields.io/github/stars/Aar2284/World_Bank_Socio_Economic_Data_Analysis)](https://github.com/Aar2284/World_Bank_Socio_Economic_Data_Analysis/stargazers)
[![Forks](https://img.shields.io/github/forks/Aar2284/World_Bank_Socio_Economic_Data_Analysis)](https://github.com/Aar2284/World_Bank_Socio_Economic_Data_Analysis/network/members)

This project analyzes and models global socio-economic indicators using World Bank data, aiming to understand patterns, disparities, and to predict economic outcomes with machine learning techniques. Recent updates include configuration for deploying interactive dashboards with Streamlit.
---

## 📁 Dataset

- Sourced from the World Bank; multiple countries over different years.
- Indicators: GDP, GDP per capita, life expectancy, internet usage, birth/death rates, regions, and income groups.
- **Files**:
  - `WorldBank.xlsx`: Original dataset.
  - `WorldBank_Cleaned.csv`: Cleaned dataset after preprocessing.

---

## Key Features & Benefits

*   **Data Exploration:** In-depth analysis of World Bank socio-economic indicators.
*   **Interactive Dashboards:** Visualize data using Streamlit for easy exploration and analysis.
*   **Predictive Modeling:** Utilizes machine learning algorithms to forecast economic outcomes.
*   **Comparative Analysis:** Compare socio-economic indicators across different countries and regions.
*   **Open Source:** Freely available for modification and extension.

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

The `requirements.txt` file includes:

*   `streamlit`: For creating interactive web applications.
*   `pandas`: For data manipulation and analysis.
*   `plotly`: For creating interactive plots and visualizations.
*   `scikit-learn`: For machine learning algorithms and tools.
*   `openpyxl`: For reading and writing Excel files.
*   `statsmodels`: For statistical modeling.

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

## Project Structure

```
World_Bank_Socio_Economic_Data_Analysis/
├── .streamlit/
│   └── config.toml       # Streamlit configuration file
├── README.md              # This file
├── WorldBank.xlsx         # Dataset (World Bank data)
├── dashboard.py           # Streamlit dashboard application
├── graphs.py              # Script for generating graphs and performing analysis
└── requirements.txt       # List of Python dependencies
```

---

## Contributing Guidelines

Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive commit messages.
4.  Submit a pull request to the main branch.

---

## Configuration Options

The Streamlit dashboard can be customized by modifying the `config.toml` file located in the `.streamlit/` directory.  This file allows you to adjust settings such as the default theme, layout, and more.  Refer to the Streamlit documentation for details on available configuration options.

---

## Acknowledgments

*   World Bank: For providing the socio-economic data.
*   Streamlit: For the awesome framework to quickly build interactive dashboards.
*   Pandas, Plotly, Scikit-learn: for their great libraries.

---

**Presentation Tip:**  
Present visuals objective by objective. Explain model logic and accuracy (R², MSE). Highlight insights such as how internet usage and life expectancy affect GDP per capita.
