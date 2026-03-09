from i_dashboard_view import IDashboardView
from sale_record import SaleRecord


class TableView(IDashboardView):
    def refresh(self) -> None:
        pass

    def render_table(self, data: list[SaleRecord]) -> None:
        if not data:
            print("No data to display.")
            return

        header = f"{'Date':<14}{'Location':<20}{'Product':<16}{'Qty':>6}"
        print(header)
        print("-" * len(header))
        for record in data:
            print(
                f"{str(record.date):<14}"
                f"{record.bakery_location:<20}"
                f"{record.product.name:<16}"
                f"{record.quantity_sold:>6}"
            )
