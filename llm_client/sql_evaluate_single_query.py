import sqlite3
import os
import pandas as pd


class SQLEvaluator:

    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        self.connections = {}

    def get_connection(self, db_id: str) -> sqlite3.Connection:
        if db_id not in self.connections:
            db_path = os.path.join(self.db_dir, f"{db_id}/{db_id}.sqlite")
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found: {db_path}")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            self.connections[db_id] = conn
        return self.connections[db_id]

    def execute_query(self, db_id: str, query: str) -> list:
        try:
            conn = self.get_connection(db_id)
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            return pd.DataFrame([dict(row) for row in rows])
        except Exception as e:
            return f"ERROR: {e}"


def evaluate_sql_query(query, retrieved_db):
    """
    Evaluate the SQL query against the SQLite database.
    """
    # Create a connection to the SQLite database
    evaluator = SQLEvaluator(db_dir="llm_client/output/dev_databases")
    print("----------------------")
    print(retrieved_db[0][0]['db'])
    print("----------------------")
    db_id = retrieved_db[0][0]['db']
    result = evaluator.execute_query(db_id=db_id, query=query)
    return result

