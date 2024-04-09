import torch
import uvicorn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from pydantic import BaseModel
import os

device = os.environ.get("DEVICE") or 'cpu'
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru").to(device)

app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/embedding")
def calc_embedding(query: Query):
    encoded_input = tokenizer(query.query, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**(encoded_input.to(device)))
    query_embedding = model_output.pooler_output
    query_embedding = torch.nn.functional.normalize(query_embedding).squeeze(0)

    return query_embedding.tolist()


if __name__ == "__main__":
    print("123")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT')))
