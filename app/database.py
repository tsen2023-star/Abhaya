from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os

# Using SQLite for instant local development. 
# When deploying, swap this to your PostgreSQL URL via environment variables.
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./abhaya_local.db")

# Render uses 'postgres://' but SQLAlchemy requires 'postgresql://'
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

# connect_args is needed only for SQLite to prevent thread issues
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args=connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get the database session in your API routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()