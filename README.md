## RAG-система, proof-of-concept

Запуск приложения
1. Запустить Milvus:
```shell
cd milvus-docker && docker-compose up -d
```
В файле ```docker-compose.yml``` вставить авторизационные данные для Gigachat в переменную окружения ```GIGACHAT``` приложения ```search```
2. Запустить приложение
```shell
cd .. && docker-compose up -d --build
```
3. Проверить запущенные контейнеры можно с помощью команды:
```shell
docker-compose ps
```
4. Выполнить скрипт ```create_collection.py``` - создание коллекции
5. Выполнить скрипт ```text_processing.py``` - заполнение коллекции данными из ```dataset.json```
6. Выполнить скрипт ```test_search.py``` - получение ответов от приложения search

Пример обращения к сервису, векторизующему текст:
```python
import requests
response = requests.post("http://localhost:8908/embedding", json={"query": "запрос"}).json()
```
Пример обращения к поисковому приложению:
```python
import requests
response = requests.post("http://localhost:8909/search", json={"query": "Запрос"}).json()
```
