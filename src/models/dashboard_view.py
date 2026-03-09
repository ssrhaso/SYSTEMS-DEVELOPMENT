from abc import ABC, abstractmethod


class IDashboardView(ABC):
    @abstractmethod
    def refresh(self) -> None:
        pass
