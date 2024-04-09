import json
import requests

with open("queries.json") as f:
    queries = json.load(f)

search_result = [
    requests.post("http://localhost:8909/search", json={"query": query}).json() for query in queries
]

with open('search_results.json', 'w', encoding='utf-8') as f:
    json.dump(search_result, f, ensure_ascii=False)
