from .category import Category


class Product:
    def __init__(self, name: str, category: Category) -> None:
        self.name: str = name
        self.category: Category = category

    def get_type(self) -> Category:
        return self.category

    def __repr__(self) -> str:
        return f"Product(name='{self.name}', category={self.category})"
