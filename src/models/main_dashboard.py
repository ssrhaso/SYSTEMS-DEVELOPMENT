from enum import Enum

import streamlit as st


class ViewType(Enum):
    TABLE = "Table"
    GRAPH = "Graph"


class MainDashboard:
    def __init__(self) -> None:
        self.current_view: ViewType = ViewType.TABLE
        self._view = None

    def display(self) -> None:
        if self._view is not None:
            print(f"Dashboard loaded — current view: {self.current_view.value}")

    def toggle_view(self, view_type: ViewType) -> None:
        self.current_view = view_type

    def set_view(self, view) -> None:
        self._view = view

    def show_error_message(self, msg: str) -> None:
        st.error(msg)
