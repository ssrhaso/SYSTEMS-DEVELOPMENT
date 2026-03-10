from datetime import date

import plotly.graph_objects as go

from .prediction import Prediction
from .sale_record import SaleRecord

PRODUCT_COLOURS = {
    "Cappuccino": "#E8547A",
    "Americano": "#BE185D",
    "Croissants": "#8B5CF6",
}


class GraphView:
    PINK = "#E8547A"

    def plot_history(self, data: list[SaleRecord]) -> go.Figure:
        if not data:
            return go.Figure()
        dates = [record.date for record in data]
        quantities = [record.quantity_sold for record in data]
        product_name = data[0].product.name
        colour = PRODUCT_COLOURS.get(product_name, self.PINK)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=quantities,
            mode="lines+markers", name=product_name,
            line=dict(color=colour, width=2),
            marker=dict(size=4),
        ))
        return fig

    def plot_prediction(self, data: list[Prediction]) -> go.Figure:
        if not data:
            return go.Figure()
        dates = [p.predicted_date for p in data]
        quantities = [p.predicted_quantity for p in data]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=quantities,
            mode="lines+markers", name="28-Day Forecast",
            line=dict(color=self.PINK, width=2.5, dash="dash"),
            marker=dict(size=5, color=self.PINK),
        ))
        return fig

    def plot_history_and_prediction(
        self, history: list[SaleRecord], predictions: list[Prediction]
    ) -> go.Figure:
        fig = go.Figure()
        if history:
            fig.add_trace(go.Scatter(
                x=[r.date for r in history],
                y=[r.quantity_sold for r in history],
                mode="lines", name="Historical",
                line=dict(color="#9CA3AF", width=2),
            ))
        if predictions:
            fig.add_trace(go.Scatter(
                x=[p.predicted_date for p in predictions],
                y=[p.predicted_quantity for p in predictions],
                mode="lines+markers", name="28-Day Forecast",
                line=dict(color=self.PINK, width=2.5, dash="dash"),
                marker=dict(size=5, color=self.PINK),
            ))
        return fig

    def plot_multi_product_history(
        self, sales_by_product: dict[str, list[SaleRecord]]
    ) -> go.Figure:
        fig = go.Figure()
        for product_name, records in sales_by_product.items():
            if not records:
                continue
            dates = [r.date for r in records]
            quantities = [r.quantity_sold for r in records]
            colour = PRODUCT_COLOURS.get(product_name, self.PINK)
            fig.add_trace(go.Scatter(
                x=dates, y=quantities,
                mode="lines+markers", name=product_name,
                line=dict(color=colour, width=2),
                marker=dict(size=4),
            ))
        return fig

    def zoom_to_range(self, start: date, end: date) -> None:
        pass  # Handled by Plotly's built-in zoom controls in Streamlit
