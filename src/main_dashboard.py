from enum import Enum

from i_dashboard_view import IDashboardView


class ViewType(Enum):
    TABLE = "Table"
    GRAPH = "Graph"


class MainDashboard:
    def __init__(self) -> None:
        self.current_view: ViewType = ViewType.TABLE
        self._view: IDashboardView | None = None

    def display(self) -> None:
        if self._view is not None:
            self._view.refresh()
        else:
            print(f"Dashboard loaded — current view: {self.current_view.value}")

    def toggle_view(self, view_type: ViewType) -> None:
        self.current_view = view_type

    def set_view(self, view: IDashboardView) -> None:
        self._view = view

    def show_error_message(self, msg: str) -> None:
        print(f"[ERROR] {msg}")
