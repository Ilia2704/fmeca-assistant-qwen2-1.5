# db_clients.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

load_dotenv()

def get_qdrant() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)

def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "1234!")
    return GraphDatabase.driver(uri, auth=(user, password))
