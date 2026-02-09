## Bakery Sales Forecasting — Bristol-Pink

### Overview

This repository contains the code and resources for a Bakery Sales Forecasting system developed as part of the Systems Development Group Project. The solution analyzes historical sales data, builds predictive models, and provides a dashboard for visualizing forecasts and insights to support inventory and purchasing decisions.

### Objectives

- Provide accurate short-term sales forecasts for key products.
- Reduce food waste and optimize inventory purchasing.
- Offer interactive visualizations to support operational decisions.

### Features

- Data ingestion and preprocessing for CSV sales data.
- Exploratory data analysis and visual summaries.
- Multiple predictive models with evaluation metrics (MAE, RMSE, R²).
- Interactive dashboard for viewing historical data and forecasts.

### Repository Structure

- `data/` — raw and processed data files (CSV).
- `src/` — data loading, preprocessing, feature engineering, and model training code.
- `models/` — model definitions and evaluation utilities.
- `Dashboard/` — dashboard application (`app.py`) and visualization code.
- `analysis/` — exploratory analysis notebooks/scripts.

### Setup

Recommended Python: 3.8 or newer.

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
# or .venv\Scripts\activate  # cmd.exe
```

2. Install dependencies (if `requirements.txt` is present):

```powershell
pip install -r requirements.txt
```

### Running the Dashboard

To run the dashboard application, open `Dashboard/app.py` and run it with the appropriate runner. If the dashboard uses Streamlit, use:

```powershell
streamlit run Dashboard/app.py
```

Otherwise run directly with Python:

```powershell
python Dashboard\app.py
```

### Data

Place raw CSV files in `data/raw/`. Example files included in this repository:

- `data/raw/Pink_CoffeeSales_March - Oct 2025.csv`
- `data/raw/Pink_CroissantSales_March-Oct_2025.csv`

### Model Training & Evaluation

Model training scripts are in `src/models/` and `models/`. Evaluation metrics (MAE, RMSE, R²) are produced by the evaluation utilities in `src/models` and `models/evaluator.py`.

### Project Contributors

- Ishaq Modassir Mushtaq
- Rayyan Tahir
- Royden Valerian Dias
- Hasaan Ahmad
- Chouaib Hakim

### License

This project was developed for the SDGP coursework. Please contact the project team for reuse or distribution permissions.

---

If you would like, I can also:

- add a `requirements.txt` generated from the environment,
- run a quick lint/format pass, or
- preview the dashboard locally and confirm startup instructions.

