from datetime import date

from product import Product


class SaleRecord:
    def __init__(self, sale_date: date, bakery_location: str, quantity_sold: int, product: Product) -> None:
        self.date: date = sale_date
        self.bakery_location: str = bakery_location
        self.quantity_sold: int = quantity_sold
        self.product: Product = product

    def get_week_number(self) -> int:
        return self.date.isocalendar()[1]

    def __repr__(self) -> str:
        return (
            f"SaleRecord(date={self.date}, location='{self.bakery_location}', "
            f"product='{self.product.name}', qty={self.quantity_sold})"
        )
