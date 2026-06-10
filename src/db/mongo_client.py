"""
MongoDB client initialization.
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

#MONGO_URL = "mongodb://localhost:27017"
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "adaptive_rag"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
