from sqlalchemy import create_engine, text
from langchain.tools import Tool
from typing import List, Dict, Any
import pandas as pd

class DatabaseTool:
    def __init__(self, connection_string: str):
        """Initialize database connection"""
        self.engine = create_engine(connection_string)
    
    def query_database(self, query: str) -> List[Dict[Any, Any]]:
        """Execute SQL query and return results"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                print(f"Executing query: {text(query)}")
                columns = result.keys()
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def get_table_schema(self, table_name: str) -> str:
        """Get schema information for a specific table"""
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 0", self.engine)
            return str(df.dtypes)
        except Exception as e:
            return f"Error getting schema: {str(e)}"