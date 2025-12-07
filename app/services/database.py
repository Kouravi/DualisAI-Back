from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DB_NAME = "ai-mongodb"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db["predictions"]

async def insert_prediction(data: dict):
    result = await collection.insert_one(data)
    return str(result.inserted_id)

def serialize_doc(doc):
    """Convierte los campos BSON (ObjectId, datetime, etc.) a tipos serializables por JSON"""
    doc["_id"] = str(doc["_id"])
    return doc

async def get_all_predictions():
    preds = await collection.find().to_list(1000)
    return [serialize_doc(p) for p in preds]