import logging
import os
import sys
import warnings
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

VALID_ALGORITHMS = {"Prophet", "ARIMA", "XGBoost", "Ensemble"}


class Category(Enum):
    COFFEE = "Coffee"
    PASTRY = "Pastry"


class AlgorithmType(Enum):
    PROPHET = "Prophet"
    ARIMA = "ARIMA"
    XGBOOST = "XGBoost"
    ENSEMBLE = "Ensemble"

    def _missing_(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None


class ViewType(Enum):
    GRAPH = "graph"
    TABLE = "table"


@dataclass
class Product:
    name: str
    category: Category

    def get_type(self) -> Category:
        return self.category


@dataclass
class SaleRecord:
    sale_date: date
    bakery_location: str
    quantity_sold: int
    product: Product

    @property
    def date(self) -> date:
        return self.sale_date


@dataclass
class Prediction:
    product: Product
    predicted_date: date
    predicted_quantity: float


class CSVReader:
    @staticmethod
    def read_rows(file_path: str) -> list[list[str]]:
        with open(file_path, "r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.reader(handle))


class DataManager:
    def __init__(self) -> None:
        self._df = pd.DataFrame()

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        if "Date" in self._df.columns:
            self._df["Date"] = pd.to_datetime(self._df["Date"], errors="coerce")

    def get_sales_by_product(self, product_name: str) -> list[SaleRecord]:
        if self._df.empty:
            return []
        if "Date" not in self._df.columns or product_name not in self._df.columns:
            return []

        category = Category.COFFEE if product_name != "Croissants" else Category.PASTRY
        product = Product(product_name, category)
        rows = self._df[["Date", product_name]].dropna()

        sales: list[SaleRecord] = []
        for _, row in rows.iterrows():
            dt = row["Date"]
            qty = pd.to_numeric(row[product_name], errors="coerce")
            if pd.isna(dt) or pd.isna(qty):
                continue
            sales.append(
                SaleRecord(
                    sale_date=pd.Timestamp(dt).date(),
                    bakery_location="Bristol Centre",
                    quantity_sold=int(max(float(qty), 0.0)),
                    product=product,
                )
            )
        return sales


class SalesAnalyzer:
    def get_sales_fluctuation(
        self,
        product: Product,
        weeks: int,
        sales: list[SaleRecord],
    ) -> dict[str, int]:
        if not sales:
            return {}

        last_seen = max(s.date for s in sales)
        start_date = last_seen - timedelta(days=max(1, int(weeks)) * 7 - 1)

        totals: dict[str, int] = {}
        for record in sales:
            if record.product.name != product.name or record.date < start_date:
                continue
            week_start = record.date - timedelta(days=record.date.weekday())
            key = week_start.isoformat()
            totals[key] = totals.get(key, 0) + int(record.quantity_sold)
        return totals


class PredictionEngine:
    def __init__(self) -> None:
        self.algorithm = AlgorithmType.PROPHET
        self.training_weeks = 4
        self._sales: list[SaleRecord] = []

    def train_model(self, sales: list[SaleRecord]) -> None:
        self._sales = list(sales)

    def set_training_period(self, train_weeks: int) -> None:
        self.training_weeks = int(train_weeks)

    def run_forecast_from_series(
        self,
        series: pd.DataFrame,
        algorithm: str | AlgorithmType,
        train_weeks: int,
    ) -> dict[str, Any]:
        algo_name = algorithm.value if isinstance(algorithm, AlgorithmType) else str(algorithm)
        return run_forecast(series, algo_name, int(train_weeks))


class AccuracyEvaluator:
    def __init__(self) -> None:
        self._accuracy_score = 0.0

    def compare_from_result(self, result: dict[str, Any]) -> None:
        metrics = (result or {}).get("metrics", {})
        mape = metrics.get("mape")
        if mape is None or (isinstance(mape, float) and np.isnan(mape)):
            self._accuracy_score = 0.0
            return
        self._accuracy_score = max(0.0, 100.0 - float(mape))

    def get_accuracy_score(self) -> float:
        return self._accuracy_score


class GraphView:
    def plot_multi_product_history(self, sales_by_product: dict[str, list[SaleRecord]]) -> go.Figure:
        fig = go.Figure()
        for product_name, records in sales_by_product.items():
            if not records:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[r.date for r in records],
                    y=[r.quantity_sold for r in records],
                    mode="lines+markers",
                    name=product_name,
                    line=dict(width=2),
                    marker=dict(size=5),
                )
            )
        return fig

    def plot_history_and_prediction(
        self,
        history_records: list[SaleRecord],
        prediction_records: list[Prediction],
    ) -> go.Figure:
        fig = go.Figure()
        if history_records:
            fig.add_trace(
                go.Scatter(
                    x=[r.date for r in history_records],
                    y=[r.quantity_sold for r in history_records],
                    mode="lines+markers",
                    name="Historical",
                    line=dict(width=2),
                    marker=dict(size=4),
                )
            )
        if prediction_records:
            fig.add_trace(
                go.Scatter(
                    x=[r.predicted_date for r in prediction_records],
                    y=[r.predicted_quantity for r in prediction_records],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(width=2, dash="dash"),
                    marker=dict(size=5),
                )
            )
        return fig


class TableView:
    def render_dataframe(self, data: pd.DataFrame) -> None:
        import streamlit as st

        st.dataframe(data, use_container_width=True, hide_index=True)


class MainDashboard:
    def __init__(self) -> None:
        self.current_view_type = ViewType.GRAPH
        self.current_view: Any = None

    def set_view(self, view: Any) -> None:
        self.current_view = view

    def toggle_view(self, view_type: ViewType) -> None:
        self.current_view_type = view_type

    def show_error_message(self, message: str) -> None:
        import streamlit as st

        st.error(message)


class DashboardController:
    def __init__(self) -> None:
        self.data_manager = DataManager()
        self.sales_analyzer = SalesAnalyzer()
        self.prediction_engine = PredictionEngine()
        self.accuracy_evaluator = AccuracyEvaluator()
        self.graph_view = GraphView()
        self.table_view = TableView()
        self.dashboard = MainDashboard()


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(
        (np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)) ** 2
    )))


def _make_features(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"y": values}, index=dates)
    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_7"] = df["y"].shift(7)
    df["rolling_mean_7"] = df["y"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df["y"].shift(1).rolling(14).mean()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    return df.dropna()


def _forecast_prophet(train: pd.DataFrame, horizon: int = 28) -> np.ndarray:
    from prophet import Prophet
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5,
    )
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            m.fit(train[["ds", "y"]])
        finally:
            sys.stdout = old_stdout
    future = m.make_future_dataframe(periods=horizon, freq="D")
    forecast = m.predict(future)
    return np.clip(forecast.tail(horizon)["yhat"].values, 0, None)


def _forecast_arima(train_y: np.ndarray, horizon: int = 28) -> np.ndarray:
    import pmdarima as pm
    model = pm.auto_arima(
        train_y,
        seasonal=True, m=7,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
    )
    yhat, _ = model.predict(n_periods=horizon, return_conf_int=True)
    return np.clip(yhat, 0, None)


def _forecast_xgboost(
    train_y: np.ndarray,
    train_dates: pd.DatetimeIndex,
    horizon: int = 28,
) -> np.ndarray:
    from xgboost import XGBRegressor

    feat_df = _make_features(train_y, train_dates)
    feature_cols = [
        "lag_1", "lag_2", "lag_7",
        "rolling_mean_7", "rolling_mean_14",
        "day_of_week", "month",
    ]

    model = XGBRegressor(
        n_estimators=200, max_depth=3,
        learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        verbosity=0, random_state=42,
    )
    model.fit(feat_df[feature_cols].values, feat_df["y"].values)

    history = list(train_y)
    last_date = train_dates[-1]
    predictions: list[float] = []

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)
        lag_1 = history[-1]
        lag_2 = history[-2] if len(history) >= 2 else history[-1]
        lag_7 = history[-7] if len(history) >= 7 else history[0]
        w7 = history[-7:] if len(history) >= 7 else history
        w14 = history[-14:] if len(history) >= 14 else history
        x = np.array([[
            lag_1, lag_2, lag_7,
            float(np.mean(w7)), float(np.mean(w14)),
            next_date.dayofweek, next_date.month,
        ]])
        pred = float(max(model.predict(x)[0], 0.0))
        predictions.append(pred)
        history.append(pred)

    return np.array(predictions)


def _forecast_ensemble(train_df: pd.DataFrame, horizon: int = 28) -> np.ndarray:
    arima = _forecast_arima(train_df["y"].values, horizon)
    xgb = _forecast_xgboost(
        train_df["y"].values,
        pd.DatetimeIndex(train_df["ds"]),
        horizon,
    )
    return np.clip(0.5 * arima + 0.5 * xgb, 0, None)


def _evaluate(
    series: pd.DataFrame,
    algorithm: str,
    train_size: int,
    test_size: int = 28,
    n_splits: int = 3,
) -> dict:
    mapes, maes, rmses = [], [], []

    for i in range(n_splits):
        start = train_size + i * test_size
        if start + test_size > len(series):
            break

        train = series.iloc[start - train_size: start].reset_index(drop=True)
        test = series.iloc[start: start + test_size]
        actual = test["y"].values

        try:
            if algorithm == "Prophet":
                yhat = _forecast_prophet(train, horizon=len(test))
            elif algorithm == "ARIMA":
                yhat = _forecast_arima(train["y"].values, horizon=len(test))
            elif algorithm == "XGBoost":
                yhat = _forecast_xgboost(
                    train["y"].values,
                    pd.DatetimeIndex(train["ds"]),
                    horizon=len(test),
                )
            elif algorithm == "Ensemble":
                yhat = _forecast_ensemble(train, horizon=len(test))
        except Exception:
            continue

        n = min(len(actual), len(yhat))
        mapes.append(_mape(actual[:n], yhat[:n]))
        maes.append(float(mean_absolute_error(actual[:n], yhat[:n])))
        rmses.append(_rmse(actual[:n], yhat[:n]))

    if not mapes:
        return {"mape": np.nan, "mae": np.nan, "rmse": np.nan}

    return {
        "mape": float(np.mean(mapes)),
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
    }


def run_forecast(series: pd.DataFrame, algorithm: str, train_weeks: int) -> dict:
    _empty = pd.DataFrame(columns=["ds", "yhat"])
    _empty_hist = pd.DataFrame(columns=["ds", "y"])
    _empty_metrics = {"mape": None, "mae": None, "rmse": None, "meets_target": False}

    def _fail(msg: str) -> dict:
        return {
            "forecast_df": _empty,
            "history_df": _empty_hist,
            "metrics": _empty_metrics,
            "error": msg,
        }

    if algorithm not in VALID_ALGORITHMS:
        return _fail(f"Unknown algorithm '{algorithm}'.")

    if not isinstance(series, pd.DataFrame) or not {"ds", "y"}.issubset(series.columns):
        return _fail("'series' must be a DataFrame with columns ['ds', 'y'].")

    try:
        series = series.copy()
        series["y"] = pd.to_numeric(series["y"], errors="raise")
    except (ValueError, TypeError):
        return _fail("Column 'y' contains non-numeric values.")

    if series["y"].isna().any():
        return _fail("Column 'y' contains NaN values.")

    train_size = int(train_weeks) * 7
    min_required = train_size + 28
    if len(series) < min_required:
        return _fail(
            f"Need at least {min_required} rows "
            f"({train_weeks} weeks x 7 + 28 test days). "
            f"Got {len(series)}."
        )

    try:
        raw = _evaluate(series, algorithm, train_size)
    except Exception as exc:
        return _fail(f"Metric computation failed: {exc}")

    def _sr(v):
        return round(float(v), 4) if v is not None and not np.isnan(v) else None

    meets = bool(
        raw["mape"] is not None
        and not np.isnan(raw["mape"])
        and raw["mape"] <= 35.0
    )
    metrics = {
        "mape": _sr(raw["mape"]),
        "mae": _sr(raw["mae"]),
        "rmse": _sr(raw["rmse"]),
        "meets_target": meets,
    }

    train_df = series.iloc[-train_size:].reset_index(drop=True)

    try:
        if algorithm == "Prophet":
            yhat_vals = _forecast_prophet(train_df, horizon=28)
        elif algorithm == "ARIMA":
            yhat_vals = _forecast_arima(train_df["y"].values, horizon=28)
        elif algorithm == "XGBoost":
            yhat_vals = _forecast_xgboost(
                train_df["y"].values,
                pd.DatetimeIndex(train_df["ds"]),
                horizon=28,
            )
        elif algorithm == "Ensemble":
            yhat_vals = _forecast_ensemble(train_df, horizon=28)
    except Exception as exc:
        return _fail(f"Forecasting failed ({algorithm}): {exc}")

    last_date = pd.Timestamp(train_df["ds"].iloc[-1])
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=28, freq="D"
    )

    return {
        "forecast_df": pd.DataFrame({"ds": forecast_dates, "yhat": np.clip(yhat_vals, 0, None)}),
        "history_df": train_df[["ds", "y"]].copy(),
        "metrics": metrics,
        "error": None,
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessor import load_all, to_series

    df = load_all()
    series = to_series(df, "Cappuccino")

    print("Testing ARIMA on Cappuccino...")
    result = run_forecast(series, "ARIMA", train_weeks=4)

    if result["error"]:
        print(f"ERROR: {result['error']}")
    else:
        print(f"Metrics : {result['metrics']}")
        print(f"Forecast:\n{result['forecast_df'].head()}")