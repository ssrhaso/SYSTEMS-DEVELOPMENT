from datetime import date

from data_manager import DataManager
from main_dashboard import MainDashboard
from prediction_engine import PredictionEngine
from sales_analyzer import SalesAnalyzer
from graph_view import GraphView
from table_view import TableView
from main_dashboard import ViewType


class DashboardController:
    def __init__(self) -> None:
        self._data_manager: DataManager = DataManager()
        self._main_dashboard: MainDashboard = MainDashboard()
        self._prediction_engine: PredictionEngine = PredictionEngine()
        self._sales_analyzer: SalesAnalyzer = SalesAnalyzer()
        self._graph_view: GraphView = GraphView()
        self._table_view: TableView = TableView()

    def select(self, path: str) -> None:
        if path.lower().endswith(".csv"):
            self._data_manager.load_drink_data(path)
        self._data_manager.merge_datasets()
        self._main_dashboard.display()

    def on_training_period_change(self, weeks: int) -> None:
        self._prediction_engine.set_training_period(weeks)

    def on_zoom(self, start_date: date, end_date: date) -> None:
        self._graph_view.zoom_to_range(start_date, end_date)

    def generate_prediction(self) -> None:
        sales = self._data_manager.get_combined_sales()
        if not sales:
            self._main_dashboard.show_error_message("No sales data loaded.")
            return

        product = sales[0].product
        self._prediction_engine.train_model(sales)
        predictions = self._prediction_engine.predict_sales(product, future_weeks=4)

        if self._main_dashboard.current_view == ViewType.GRAPH:
            self._graph_view.plot_prediction(predictions)
        else:
            for p in predictions:
                print(p)
