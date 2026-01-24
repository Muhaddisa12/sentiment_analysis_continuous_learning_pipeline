"""
Data Loading Module

Loads training data from MySQL database using configuration from config.py.
"""

from sqlalchemy import create_engine
import pandas as pd
from config import DB_CONFIG


def get_engine():
    """Create SQLAlchemy engine from configuration."""
    user = DB_CONFIG["user"]
    password = DB_CONFIG["password"]
    host = DB_CONFIG["host"]
    database = DB_CONFIG["database"]
    
    connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
    return create_engine(connection_string)


def load_database():
    """
    Load training data from database.
    
    Returns:
        pandas.DataFrame: Training data with columns including 'clean_text' and 'category'
    """
    engine = get_engine()
    query = "SELECT * FROM twitter_data"
    df = pd.read_sql(query, engine)
    return df
