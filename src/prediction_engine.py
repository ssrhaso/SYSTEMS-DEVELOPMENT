from datetime import timedelta
from enum import Enum

from prediction import Prediction
from product import Product
from sale_record import SaleRecord


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

    def set_training_period(self, weeks: int) -> None:
        self.training_period_weeks = weeks

    def train_model(self, history_data: list[SaleRecord]) -> None:
        cutoff = self.training_period_weeks * 7
        self._history_data = history_data[-cutoff:] if len(history_data) > cutoff else list(history_data)

    def predict_sales(self, product: Product, future_weeks: int) -> list[Prediction]:
        if not self._history_data:
            return []

        total_qty = sum(r.quantity_sold for r in self._history_data)
        days_count = len(self._history_data) if self._history_data else 1
        avg_daily = total_qty / days_count

        last_date = max(r.date for r in self._history_data)
        predictions: list[Prediction] = []
        for day_offset in range(1, future_weeks * 7 + 1):
            predicted_date = last_date + timedelta(days=day_offset)
            predictions.append(
                Prediction(
                    product=product,
                    predicted_date=predicted_date,
                    predicted_quantity=round(avg_daily, 1),
                )
            )
        return predictions
