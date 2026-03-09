import streamlit as st
import pandas as pd

from .i_dashboard_view import IDashboardView
from .sale_record import SaleRecord


class TableView(IDashboardView):
    def refresh(self) -> None:
        pass

    def render_table(self, data: list[SaleRecord]) -> None:
        if not data:
            st.info("No data to display.")
            return
        rows = []
        for record in data:
            rows.append({
                "Date": str(record.date),
                "Location": record.bakery_location,
                "Product": record.product.name,
                "Category": record.product.get_type().value,
                "Qty Sold": record.quantity_sold,
            })
        self._render_html_table(pd.DataFrame(rows))

    def render_dataframe(self, data: pd.DataFrame) -> None:
        if data.empty:
            st.info("No data to display.")
            return
        self._render_html_table(data)

    @staticmethod
    def _render_html_table(data: pd.DataFrame) -> None:
        header_cells = "".join(
            f"<th style='background:#F3F4F6;color:#111827;font-weight:600;"
            f"padding:0.55rem 0.85rem;text-align:left;font-size:0.83rem;"
            f"border-bottom:2px solid #E5E7EB;white-space:nowrap;'>{col}</th>"
            for col in data.columns
        )
        rows_html = ""
        for i, row in enumerate(data.itertuples(index=False)):
            bg = "#FFFFFF" if i % 2 == 0 else "#F9FAFB"
            cells = "".join(
                f"<td style='padding:0.48rem 0.85rem;color:#111827;"
                f"font-size:0.83rem;border-bottom:1px solid #F3F4F6;"
                f"white-space:nowrap;'>{v}</td>"
                for v in row
            )
            rows_html += f"<tr style='background:{bg};'>{cells}</tr>"
        st.markdown(
            f"<div style='overflow-x:auto;border:1px solid #E5E7EB;"
            f"border-radius:8px;margin-top:0.4rem;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr>{header_cells}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>",
            unsafe_allow_html=True,
        )
