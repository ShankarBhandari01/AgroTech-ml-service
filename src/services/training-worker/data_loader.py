import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://user:pass@localhost/db"

def load_training_data():
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT *
    FROM training_dataset
    """

    return pd.read_sql(query, engine)