from datetime import date

from .product import Product


class Prediction:
    def __init__(self, product: Product, predicted_date: date, predicted_quantity: float) -> None:
        self.product: Product = product
        self.predicted_date: date = predicted_date
        self.predicted_quantity: float = predicted_quantity

    def __repr__(self) -> str:
        return (
            f"Prediction(product={self.product.name}, "
            f"date={self.predicted_date}, qty={self.predicted_quantity:.1f})"
        )
