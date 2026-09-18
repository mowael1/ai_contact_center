import pyodbc

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# We made it false to make SQLAlchemy Responsible for connection pooling
pyodbc.pooling = False

odbc_connection_string = (
    f"DRIVER={{{settings.DB_DRIVER}}};"
    f"SERVER={settings.DB_SERVER};"
    f"DATABASE={settings.DB_NAME};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

connection_url = URL.create(
    "mssql+pyodbc",
    query={
        "odbc_connect": odbc_connection_string
    }
)

engine = create_engine(
    connection_url,
    pool_pre_ping=True,
    # echo=True
    echo=False
    
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)