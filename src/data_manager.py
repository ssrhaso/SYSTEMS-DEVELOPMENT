from datetime import datetime

from csv_reader import CSVReader
from category import Category
from product import Product
from sale_record import SaleRecord


class DataManager:
    def __init__(self) -> None:
        self.all_sales: list[SaleRecord] = []
        self._food_sales: list[SaleRecord] = []
        self._drink_sales: list[SaleRecord] = []

    def load_food_data(self, csv_path: str) -> None:
        rows = CSVReader.read_rows(csv_path)
        if not rows:
            return
        header = rows[0]
        for row in rows[1:]:
            if len(row) < len(header):
                continue
            try:
                sale_date = datetime.strptime(row[0].strip(), "%d/%m/%Y").date()
            except ValueError:
                continue
            for col_idx in range(1, len(header)):
                product_name = header[col_idx].strip()
                product = Product(name=product_name, category=Category.PASTRY)
                try:
                    quantity = int(float(row[col_idx].strip()))
                except (ValueError, IndexError):
                    quantity = 0
                record = SaleRecord(
                    sale_date=sale_date,
                    bakery_location="Bristol Centre",
                    quantity_sold=quantity,
                    product=product,
                )
                self._food_sales.append(record)

    def load_drink_data(self, csv_path: str) -> None:
        rows = CSVReader.read_rows(csv_path)
        if not rows:
            return
        header = rows[0]
        for row in rows[1:]:
            if len(row) < len(header):
                continue
            try:
                sale_date = datetime.strptime(row[0].strip(), "%d/%m/%Y").date()
            except ValueError:
                continue
            for col_idx in range(1, len(header)):
                product_name = header[col_idx].strip()
                product = Product(name=product_name, category=Category.COFFEE)
                try:
                    quantity = int(float(row[col_idx].strip()))
                except (ValueError, IndexError):
                    quantity = 0
                record = SaleRecord(
                    sale_date=sale_date,
                    bakery_location="Bristol Centre",
                    quantity_sold=quantity,
                    product=product,
                )
                self._drink_sales.append(record)

    def merge_datasets(self) -> None:
        self.all_sales = self._food_sales + self._drink_sales

    def get_combined_sales(self) -> list[SaleRecord]:
        return list(self.all_sales)

    def get_sales_by_product(self, product_name: str) -> list[SaleRecord]:
        return [
            record for record in self.all_sales
            if record.product.name == product_name
        ]
