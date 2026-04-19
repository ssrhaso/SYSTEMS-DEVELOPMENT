# Bristol Pink Cafe - Sales Forecasting Dashboard

An intelligent forecasting dashboard that predicts bakery product sales across five Bristol Pink Cafe locations using time series and machine learning algorithms, enabling data-driven inventory management to minimise food waste.

Developed as part of **UFCF7S-30-2 Systems Development Group Project** at UWE Bristol.

---

## Problem

Bakeries waste between 5–15% of daily production due to inaccurate demand estimation. Bristol Pink Cafe needed a system to forecast demand for perishable products, aligning baking volumes with actual sales to reduce waste and improve profitability.

## Solution

A Streamlit-based dashboard that ingests historical sales CSV data, visualises trends across products and locations, and generates 28-day sales predictions using configurable forecasting models — all accessible to non-technical bakery managers.

## Key Features

| Feature                           | Description                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **CSV Import & Validation** | Robust upload with format detection, error handling for malformed data, and clear user feedback (FR1) |
| **Sales Visualisation**     | Interactive line charts and bar charts with adjustable time windows - 1, 4, or 12 weeks (FR2, FR3)     |
| **Forecasting Engine**      | Prophet, ARIMA, XGBoost, and Ensemble models producing 28-day predictions with MAPE ≤ 35% (FR4)      |
| **Training Window Control** | Slider to adjust training period between 4–8 weeks, with live metric recalculation (FR5)             |
| **Data Export**             | Download forecast results as CSV and chart images as PNG (FR6)                                        |

## Repository Structure

```
├── src/
│   ├── app.py              # Streamlit dashboard (entry point)
│   ├── model.py            # Forecasting engine
│   ├── preprocessor.py     # Data loading and cleaning utilities
│   └── models/             # Domain classes and evaluation utilities
└── data/
    ├── raw/                # Historical sales CSVs
    └── processed/          # Generated chart PNGs
```

## Prerequisites

- **Python 3.10+** installed and available on `PATH`
- ~500 MB free disk space for dependencies (Prophet, XGBoost, Streamlit, etc.)
- A modern web browser (Chrome, Edge, Firefox) to view the dashboard

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
python -m streamlit run src/app.py
```

The dashboard opens at `http://localhost:8501`. Upload a CSV from `data/raw/` to begin.

## Forecasting Performance (Validated on provided dataset)

| Algorithm | Cappuccino MAPE | Americano MAPE | Croissant MAPE |
| --------- | :-------------: | :------------: | :------------: |
| Prophet   |      ~23%      |      ~19%      |      ~50%      |
| ARIMA     |      ~17%      |      ~11%      |      ~48%      |
| XGBoost   |      ~17%      |      ~14%      |      ~47%      |
| Ensemble  |      ~17%      |      ~12%      |      ~46%      |

**Values above are from the current implementation using a 4-week training window and 3-split walk-forward validation.**

> NOTE: The Performance was tested on the given dataset from our module team, and should be taken as a guideline for generalised performance, not specific.

**Croissant series shows high volatility and materially higher forecast error across all tested algorithms.**

## Contributors

| Name                   | Main Contributions                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| Hasaan Ahmad           | Project Manager, Forecasting algorithms, Pseudocode, System architecture, Streamlit App Development |
| Chouaib Hakim          | Literature review (section on food waste reduction), Use case Diagram                               |
| Ishaq Modassir Mushtaq | Data Preprocessing, Class diagrams, Unit Test Cases, ML for Demand Prediction                       |
| Rayyan Tahir           | Software engineering practices, Sequence Diagrams, Testing                                          |
| Royden Valerian Dias   | UI/UX wireframes, Dashboard Design, Streamlit App Development                                       |

## License

Developed for UFCF7S-30-2 coursework at UWE Bristol. Contact the project team for reuse or distribution permissions.
