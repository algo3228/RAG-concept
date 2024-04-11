import uuid
import requests
import json
from pymilvus import MilvusClient
from langchain.text_splitter import RecursiveCharacterTextSplitter


EMBEDDER_ADDRESS = 'http://localhost:8908/embedding'
client = MilvusClient(
    uri="http://localhost:19530"
)
COLLECTION_NAME = "LaBSE_embeddings_3"
client.load_collection(COLLECTION_NAME)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=200)


def add_text(text: str):
    texts = text_splitter.split_text(text)
    for chunk in texts:
        embedding = requests.post(EMBEDDER_ADDRESS, json={"query": chunk}).json()
        client.insert(
            collection_name=COLLECTION_NAME,
            data={
                "id": str(uuid.uuid4()),
                "text": chunk,
                "vector": embedding
            },
        )


if __name__ == '__main__':
    with open("dataset.json") as f:
        dataset = json.load(f)
    for i, item in enumerate(dataset):
        print(f"{i}/{len(dataset)}\n", item["text"])
        add_text(item["text"])
