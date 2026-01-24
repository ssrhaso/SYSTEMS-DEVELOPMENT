## Systems Development Group Project

# 🍞 SDGP – Bakery Sales Prediction System

## 📘 Project Overview

**Bristol-Pink** is a new bakery chain operating five cafés located near schools and office areas. The company caters mainly to families and local office workers.
To reduce **food waste** and improve **inventory management**, Bristol-Pink aims to use **AI and Machine Learning** to predict daily sales volumes at each café.
This project focuses on building a **standalone electronic dashboard** that analyzes past sales data, forecasts future demand, and visualizes results interactively.

---

## Project Aim

Develop an intelligent dashboard that:

- Analyzes historical sales data from CSV files.
- Identifies sales trends and top-performing products.
- Predicts future sales for high-impact items (foods and coffees).
- Provides interactive, visual insights for decision-making.

Ultimately, the system will help Bristol-PPink minimize waste, optimize purchasing, and improve operational efficiency.

---

## 🧠 Core System Requirements

### 1. Data Input & Processing

- Import and process **CSV data files** containing historical sales information.
- Perform data cleaning, formatting, and preparation for machine learning analysis.

### 2. Exploratory Data Analysis

- Identify the **top three selling food items** and **top three selling coffees**.
- Visualize **sales fluctuations** over the past four weeks using clear, interactive graphs.

### 3. Predictive Modelling

- Apply suitable **AI/ML algorithms** (e.g., Linear Regression, Random Forest, ARIMA, or LSTM).
- Train models using historical data to **forecast sales for the next four weeks**.
- Display predictions per product in individual charts.

### 4. Interactive Dashboard Features

- **Zoom and focus** on selected date ranges for detailed views.
- **Adjust the training window** (4–8 weeks) to compare model performance.
- Display results in both **tabular** and **graphical** formats.

### 5. Model Evaluation & Comparison

- Include an optional view showing **accuracy metrics** (e.g., MAE, RMSE, R²).
- Compare algorithm performance and visualize results.

---

## 🧩 Expected Deliverables

1. A **functional standalone dashboard** (desktop or web-based).
2. Interactive **data analysis and visualization** tools.
3. Accurate **sales forecasts** for the next four weeks.
4. A **model performance evaluation** section.
5. Support for **CSV data import**, user interaction, and comparison of predictions.

---

## ⚙️ Technical Stack (Suggested)

| Component                      | Technology Options                                        |
| ------------------------------ | --------------------------------------------------------- |
| **Frontend/UI**          | Python (Streamlit, Dash) or Web (React + Flask/FastAPI)   |
| **Data Processing & ML** | Python (Pandas, NumPy, Scikit-learn, TensorFlow, Prophet) |
| **Visualization**        | Matplotlib, Plotly, Seaborn                               |
| **Model Evaluation**     | MAE, MSE, RMSE, R² Score                                 |

---

## 👥 Team Roles (Suggested)

| Role                         | Responsibilities                                                      |
| ---------------------------- | --------------------------------------------------------------------- |
| **Data Engineer**      | Handle data ingestion, cleaning, and preprocessing.                   |
| **ML Specialist**      | Build, train, and tune predictive models.                             |
| **Data Analyst**       | Explore trends, generate visual insights, and assist with EDA.        |
| **Frontend Developer** | Develop dashboard UI and interactivity.                               |
| **Systems Integrator** | Combine components, perform testing, and evaluate system performance. |

---

## 🗓️ Development Roadmap (Example)

| Phase                                            | Duration   | Key Tasks                                                        |
| ------------------------------------------------ | ---------- | ---------------------------------------------------------------- |
| **Phase 1: Research & Data Understanding** | Week 1     | Collect sample data, understand structure, identify key metrics. |
| **Phase 2: Data Processing & Cleaning**    | Week 2     | Prepare and clean CSVs, handle missing data and outliers.        |
| **Phase 3: Model Development**             | Weeks 3–4 | Experiment with ML algorithms, evaluate accuracy.                |
| **Phase 4: Dashboard Design**              | Week 5     | Build dashboard UI, integrate visualization and interactivity.   |
| **Phase 5: Integration & Testing**         | Week 6     | Combine modules, test predictions, refine user experience.       |
| **Phase 6: Final Evaluation & Report**     | Week 7     | Validate performance, finalize documentation, and deploy.        |

---

## 📊 Expected Outcome

The final system will:

- Accurately **forecast café sales** for key items.
- Present **visual insights** for better decision-making.
- Allow **dynamic user interaction** (zooming, adjusting training periods).
- Support **model comparison** and algorithm evaluation.
- Contribute to Bristol-Pink’s **goal of reducing food waste** and **increasing efficiency**.

---

## 🧾 License

This project is developed as part of the **SDGP (Software Development Group Project)** coursework.
All rights reserved © 2025 – *Team Bristol-Pink AI Forecasting*.

---

## 🚀 Authors

**Team Members:**

- Ishaq Modassir Mushtaq
- Rayyan Tahir
- Royden Valerian Dias
- Hasaan Ahmad
- Chouaib Hakim 
---
