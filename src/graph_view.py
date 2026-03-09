from datetime import date

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from i_dashboard_view import IDashboardView
from prediction import Prediction
from sale_record import SaleRecord


class GraphView(IDashboardView):
    def refresh(self) -> None:
        pass

    def plot_history(self, data: list[SaleRecord]) -> None:
        if not data:
            return
        dates = [record.date for record in data]
        quantities = [record.quantity_sold for record in data]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, quantities, marker="o", linewidth=2, label="Historical Sales")
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.set_title("Sales History")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_prediction(self, data: list[Prediction]) -> None:
        if not data:
            return
        dates = [p.predicted_date for p in data]
        quantities = [p.predicted_quantity for p in data]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, quantities, marker="s", linewidth=2, linestyle="--", color="orange", label="Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Units")
        ax.set_title("Sales Prediction")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.legend()
        plt.tight_layout()
        plt.show()

    def zoom_to_range(self, start: date, end: date) -> None:
        ax = plt.gca()
        ax.set_xlim(start, end)
        plt.draw()
