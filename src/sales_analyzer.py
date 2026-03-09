from product import Product
from sale_record import SaleRecord


class SalesAnalyzer:
    def get_sales_fluctuation(self, product: Product, weeks: int) -> dict:
        return {}

    def _calculate_fluctuation(self, sales: list[SaleRecord], weeks: int) -> dict:
        weekly_totals: dict[int, int] = {}
        for record in sales:
            week_num = record.get_week_number()
            weekly_totals[week_num] = weekly_totals.get(week_num, 0) + record.quantity_sold

        sorted_weeks = sorted(weekly_totals.keys())
        recent = sorted_weeks[-weeks:] if len(sorted_weeks) >= weeks else sorted_weeks

        result: dict[int, int] = {}
        for w in recent:
            result[w] = weekly_totals[w]
        return result
