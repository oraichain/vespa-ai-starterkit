#
# https://github.com/vespa-engine/sample-apps/blob/master/news/src/python/user_search.py
# https://docs.vespa.ai/en/tutorials/news-5-recommendation.html
#

# pip install pyvespa
import time
import pandas as pd
from vespa.application import Vespa
from vespa.io import VespaResponse, VespaQueryResponse


def display_hits_as_df(response: VespaQueryResponse, fields=["doc_id", "title", "text"]) -> pd.DataFrame:
    records = []
    for hit in response.hits:
        record = {}
        for field in fields:
            record[field] = hit["fields"][field]
        records.append(record)
    return pd.DataFrame(records)


def keyword_search(app, search_query):
    query = {
        "yql": "select * from sources * where userQuery() limit 5",
        "query": search_query,
        "ranking": "bm25",
    }
    response = app.query(query)
    return display_hits_as_df(response)


def semantic_search(app, query):
    query = {
        "yql": "select * from sources * where ({targetHits:100}nearestNeighbor(embedding,e)) limit 5",
        "query": query,
        "ranking": "semantic",
        "input.query(e)": "embed(@query)",
    }
    response = app.query(query)
    return display_hits_as_df(response)


def get_text_and_embedding(doc_id):
    query = {
        "yql": f"select title, embedding from content.doc where doc_id contains '{doc_id}'",
        "hits": 1,
    }
    result = app.query(query)

    if result.hits:
        return result.hits[0]["fields"]["title"], result.hits[0]["fields"]["embedding"]
    return "", None


def query_movies_by_embedding(embedding_vector):
    query = {
        "hits": 5,
        "yql": "select * from content.doc where ({targetHits:5}nearestNeighbor(embedding, user_embedding))",
        "ranking.features.query(user_embedding)": str(embedding_vector),
        "ranking.profile": "recommendation",
    }
    results = app.query(query)
    return display_hits_as_df(results)


# Replace with the host and port of your local Vespa instance
app = Vespa(url="http://localhost", port=8080)

# query, emb = get_text_and_embedding("767")
query = "crypto, trading, tokens"
print(f"query: {query}")

# start = time.perf_counter()
# df = keyword_search(app, query)
# print(f"keyword_search: {(time.perf_counter() - start)*1000} ms")
# print(df.head(2))

start = time.perf_counter()
df = semantic_search(app, query)
print(f"semantic_search: {(time.perf_counter() - start)*1000} ms")
print(df.head(4).to_csv())

# start = time.perf_counter()
# df = query_movies_by_embedding(emb)
# print(f"embedding_search: {(time.perf_counter() - start)*1000} ms")
# print(df.head(2))
