from datetime import timedelta
from enum import Enum

import pandas as pd

from model import run_forecast
from .prediction import Prediction
from .product import Product
from .sale_record import SaleRecord


class AlgorithmType(Enum):
    PROPHET = "Prophet"
    ARIMA = "ARIMA"
    XGBOOST = "XGBoost"
    ENSEMBLE = "Ensemble"


class PredictionEngine:
    def __init__(self) -> None:
        self.training_period_weeks: int = 4
        self.algorithm: AlgorithmType = AlgorithmType.ARIMA
        self._history_data: list[SaleRecord] = []
        self._last_result: dict | None = None

    def set_training_period(self, weeks: int) -> None:
        self.training_period_weeks = weeks

    def train_model(self, history_data: list[SaleRecord]) -> None:
        cutoff = self.training_period_weeks * 7
        self._history_data = history_data[-cutoff:] if len(history_data) > cutoff else list(history_data)

    def predict_sales(self, product: Product, future_weeks: int) -> list[Prediction]:
        if not self._history_data:
            return []

        product_sales = sorted(
            [r for r in self._history_data if r.product.name == product.name],
            key=lambda r: r.date,
        )
        if not product_sales:
            return []

        series_df = pd.DataFrame({
            "ds": [r.date for r in product_sales],
            "y": [r.quantity_sold for r in product_sales],
        })
        series_df["ds"] = pd.to_datetime(series_df["ds"])

        result = run_forecast(series_df, self.algorithm.value, self.training_period_weeks)
        self._last_result = result

        if result["error"]:
            return []

        predictions: list[Prediction] = []
        for _, row in result["forecast_df"].iterrows():
            pred_date = row["ds"].date() if hasattr(row["ds"], "date") else row["ds"]
            predictions.append(Prediction(
                product=product,
                predicted_date=pred_date,
                predicted_quantity=float(row["yhat"]),
            ))
        return predictions

    def get_last_result(self) -> dict | None:
        return self._last_result

    def run_forecast_from_series(self, series: pd.DataFrame, algorithm: str, train_weeks: int) -> dict:
        result = run_forecast(series, algorithm, train_weeks)
        self._last_result = result
        return result
