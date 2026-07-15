from pymongo import MongoClient
import os

# Your MongoDB connection string.
# Render sets DATABASE_URL, which should contain your mongodb+srv:// URI.
MONGODB_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGO_INITDB_DATABASE", "abhaya_db")

# Initialize the MongoClient
client = MongoClient(MONGODB_URL)
db_instance = client[DATABASE_NAME]

def get_db():
    """
    Dependency injection for FastAPI routes.
    Returns the MongoDB database instance.
    """
    yield db_instance