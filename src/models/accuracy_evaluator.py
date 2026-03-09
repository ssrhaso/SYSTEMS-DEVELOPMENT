from .prediction import Prediction
from .sale_record import SaleRecord


class AccuracyEvaluator:
    def __init__(self) -> None:
        self._accuracy_score: float = 0.0

    def compare(self, actual: list[SaleRecord], predicted: list[Prediction]) -> None:
        if not actual or not predicted:
            self._accuracy_score = 0.0
            return

        n = min(len(actual), len(predicted))
        total_error = 0.0
        total_actual = 0.0

        for i in range(n):
            actual_qty = actual[i].quantity_sold
            predicted_qty = predicted[i].predicted_quantity
            total_error += abs(actual_qty - predicted_qty)
            total_actual += abs(actual_qty)

        if total_actual == 0:
            self._accuracy_score = 0.0
        else:
            mape = (total_error / total_actual) * 100
            self._accuracy_score = max(0.0, 100.0 - mape)

    def compare_from_result(self, result: dict) -> None:
        if result.get("error") or not result.get("metrics"):
            self._accuracy_score = 0.0
            return
        mape = result["metrics"].get("mape")
        if mape is not None:
            self._accuracy_score = max(0.0, 100.0 - mape)
        else:
            self._accuracy_score = 0.0

    def get_accuracy_score(self) -> float:
        return self._accuracy_score
