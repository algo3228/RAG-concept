from pymilvus import MilvusClient, DataType

EMBEDDINGS_DIM = 768
METRIC_TYPE = "COSINE"

# Создаем клиент
client = MilvusClient(
    uri="http://localhost:19530"
)

# Описывваем поля коллекции (как в таблице в БД). ВАЖНО! В поле vector указать Вашу размерность эмбеддинга
schema = MilvusClient.create_schema(
    auto_id=False,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDINGS_DIM)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=10000)

collection_name = "LaBSE_embeddings_2"

# Задаем поисковый индекс по векторам
index_params = MilvusClient.prepare_index_params()
# ВАЖНО! В поле metric_type указать правильную метрику
index_params.add_index(
    field_name="vector",
    metric_type=METRIC_TYPE,
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

if not client.has_collection(collection_name):
    # Создаем коллекцию
    client.create_collection(
        collection_name=collection_name, 
        schema=schema,
        index_params=index_params
    )
    print(f"Коллекция {collection_name} создана успешно.")


client.create_index(
    collection_name=collection_name,
    index_params=index_params
)
