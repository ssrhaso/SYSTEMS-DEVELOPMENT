from datetime import date

from .accuracy_evaluator import AccuracyEvaluator
from .data_manager import DataManager
from .graph_view import GraphView
from .main_dashboard import MainDashboard, ViewType
from .prediction_engine import PredictionEngine
from .sales_analyzer import SalesAnalyzer
from .table_view import TableView


class DashboardController:
    def __init__(self) -> None:
        self._data_manager: DataManager = DataManager()
        self._main_dashboard: MainDashboard = MainDashboard()
        self._prediction_engine: PredictionEngine = PredictionEngine()
        self._sales_analyzer: SalesAnalyzer = SalesAnalyzer()
        self._graph_view: GraphView = GraphView()
        self._table_view: TableView = TableView()
        self._accuracy_evaluator: AccuracyEvaluator = AccuracyEvaluator()

    @property
    def data_manager(self) -> DataManager:
        return self._data_manager

    @property
    def dashboard(self) -> MainDashboard:
        return self._main_dashboard

    @property
    def prediction_engine(self) -> PredictionEngine:
        return self._prediction_engine

    @property
    def sales_analyzer(self) -> SalesAnalyzer:
        return self._sales_analyzer

    @property
    def graph_view(self) -> GraphView:
        return self._graph_view

    @property
    def table_view(self) -> TableView:
        return self._table_view

    @property
    def accuracy_evaluator(self) -> AccuracyEvaluator:
        return self._accuracy_evaluator

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
