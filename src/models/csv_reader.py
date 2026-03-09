import csv


class CSVReader:
    @staticmethod
    def read_rows(file_path: str) -> list[list[str]]:
        rows: list[list[str]] = []
        with open(file_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        return rows
