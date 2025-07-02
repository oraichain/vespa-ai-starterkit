#
# https://github.com/vespa-engine/sample-apps/blob/master/news/src/python/user_search.py
# https://docs.vespa.ai/en/tutorials/news-5-recommendation.html
#

# pip install pyvespa
import time
import pandas as pd
import re
import requests
import json
from vespa.application import Vespa
from vespa.io import VespaResponse, VespaQueryResponse


def get_embeddings(texts, embedding_url="http://210.211.99.160:8282/embed"):
    """Get embeddings from external API"""
    try:
        response = requests.post(
            embedding_url,
            json={"inputs": texts},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None


def display_hits_as_df(
    response: VespaQueryResponse,
    fields=["conversationId", "rawContent", "hashtags", "cashtags"],
) -> pd.DataFrame:
    records = []
    for hit in response.hits:
        record = {}
        for field in fields:
            if field in hit["fields"]:
                record[field] = hit["fields"][field]
            else:
                record[field] = "N/A"
        records.append(record)
    return pd.DataFrame(records)


def get_all_documents(app, batch_size=400):
    """Get all documents from Vespa in batches"""
    all_docs = []
    offset = 0

    while True:
        query = {
            "yql": f"select * from twitter_all_tweets where true",
            "hits": batch_size,
            "offset": offset,
            "ranking": "bm25",
        }

        response = app.query(query)

        if not response.hits:
            break

        batch_docs = []
        for hit in response.hits:
            doc = {
                "conversationId": hit["fields"].get("conversationId", ""),
                "rawContent": hit["fields"].get("rawContent", ""),
                "hashtags": hit["fields"].get("hashtags", []),
                "cashtags": hit["fields"].get("cashtags", []),
            }
            batch_docs.append(doc)

        all_docs.extend(batch_docs)
        print(f"Retrieved {len(batch_docs)} documents (offset: {offset})")

        # If we got fewer documents than requested, we've reached the end
        if len(batch_docs) < batch_size:
            break

        offset += batch_size

    return pd.DataFrame(all_docs)


def count_crypto_related_tweets(app, crypto_keywords=None, use_semantic=False):
    """Count tweets related to crypto/DeFi/web3 using keyword or semantic matching"""
    if crypto_keywords is None:
        crypto_keywords = [
            "cryptocurrency",
            # "bitcoin",
            # "ethereum",
            # "blockchain",
            # "DeFi",
            # "Web3",
            # "NFT",
            # "solana",
            # "crypto",
            # "defi",
            # "web3",
            # "nft",
            # "BTC",
            # "ETH",
            # "SOL",
        ]

    crypto_counts = {}
    all_crypto_docs = []
    seen_ids = set()  # Track unique document IDs

    print("Searching for crypto-related content...")
    print(f"Keywords: {', '.join(crypto_keywords[:10])}...")
    print(f"Using {'semantic' if use_semantic else 'keyword'} search")

    for keyword in crypto_keywords:
        print(f"Searching for '{keyword}'...", end=" ")

        # Get all results for this keyword using pagination
        offset = 0
        hits_per_batch = 400  # Stay within Vespa's limit
        keyword_docs = []

        while True:
            # Vespa has a configured offset limit of 1000
            if offset >= 1000:
                break

            if use_semantic:
                # Semantic search using external embedding API
                print(f"Getting embedding for '{keyword}'...", end=" ")
                embeddings = get_embeddings([keyword])
                if embeddings is None or len(embeddings) == 0:
                    print(f"Failed to get embedding, skipping '{keyword}'")
                    continue

                query_embedding = embeddings[0]
                query = {
                    "yql": "select * from twitter_all_tweets where ({targetHits:100}nearestNeighbor(content_embedding,query_embedding))",
                    "ranking": "semantic",
                    "ranking.features.query(query_embedding)": str(query_embedding),
                    "hits": hits_per_batch,
                    "offset": offset,
                }
            else:
                # Keyword-based search
                query = {
                    "yql": f"select * from twitter_all_tweets where rawContent contains '{keyword}' or hashtags contains '{keyword}' or cashtags contains '{keyword}'",
                    "hits": hits_per_batch,
                    "offset": offset,
                    "ranking": "bm25",
                }

            try:
                response = app.query(query)
                batch_hits = response.hits

                if not batch_hits:
                    break

                # Process this batch
                for hit in batch_hits:
                    conversation_id = hit["fields"].get("conversationId", "")
                    raw_content = hit["fields"].get("rawContent", "").lower()
                    hashtags = hit["fields"].get("hashtags", [])
                    cashtags = hit["fields"].get("cashtags", [])

                    # For semantic search, add relevance filtering
                    if use_semantic:
                        relevance_score = hit.get("relevance", 0)
                        if relevance_score < 0.1:  # Skip very low relevance results
                            continue

                    if conversation_id and conversation_id not in seen_ids:
                        keyword_docs.append(
                            {
                                "conversationId": conversation_id,
                                "rawContent": hit["fields"].get("rawContent", ""),
                                "hashtags": hashtags,
                                "cashtags": cashtags,
                                "matched_keyword": keyword,
                                "relevance": hit.get("relevance", 0),
                            }
                        )
                        seen_ids.add(conversation_id)

                # If we got fewer hits than requested, we've reached the end
                if len(batch_hits) < hits_per_batch:
                    break

                offset += hits_per_batch

            except Exception as e:
                print(f"Error searching for '{keyword}': {e}")
                break

        count = len(keyword_docs)
        crypto_counts[keyword] = count
        all_crypto_docs.extend(keyword_docs)

        print(f"found {count} tweets")

    # Remove duplicates across different keywords (a tweet might match multiple keywords)
    unique_crypto_docs = []
    final_seen_ids = set()
    for doc in all_crypto_docs:
        if doc["conversationId"] not in final_seen_ids:
            unique_crypto_docs.append(doc)
            final_seen_ids.add(doc["conversationId"])

    total_crypto_tweets = len(unique_crypto_docs)

    return crypto_counts, total_crypto_tweets, pd.DataFrame(unique_crypto_docs)


def analyze_crypto_content(app, sample_size=100, use_semantic=False):
    """Analyze crypto content with detailed breakdown"""
    print("=== CRYPTO CONTENT ANALYSIS ===\n")

    # Get crypto-related tweets
    crypto_counts, total_crypto, crypto_df = count_crypto_related_tweets(
        app, use_semantic=use_semantic
    )

    # Get total document count using a simple count query
    try:
        total_query = {
            "yql": "select conversationId from twitter_all_tweets where true",
            "hits": 1,  # We just want to trigger the query
            "ranking": "bm25",
        }
        total_response = app.query(total_query)

        # Try to get total count from response
        total_documents = (
            total_response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
        )

        # If totalCount is not available, estimate from a sample
        if total_documents == 0:
            sample_query = {
                "yql": "select conversationId from twitter_all_tweets where true",
                "hits": 400,
                "ranking": "bm25",
            }
            sample_response = app.query(sample_query)
            if len(sample_response.hits) == 400:
                total_documents = "400+ (estimated)"
            else:
                total_documents = len(sample_response.hits)
    except:
        total_documents = "Unable to determine"

    print(f"📊 SUMMARY:")
    print(f"Total documents in Vespa: {total_documents}")
    print(f"Crypto-related tweets: {total_crypto}")

    if isinstance(total_documents, int) and total_documents > 0:
        print(f"Percentage: {(total_crypto/total_documents*100):.2f}%\n")
    else:
        print(f"Percentage: Unable to calculate\n")

    print(f"📈 TOP CRYPTO KEYWORDS:")
    sorted_counts = sorted(crypto_counts.items(), key=lambda x: x[1], reverse=True)
    for keyword, count in sorted_counts[:15]:
        if count > 0:
            print(f"  {keyword}: {count}")

    print(f"\n🔍 SAMPLE CRYPTO TWEETS (first {min(sample_size, len(crypto_df))}):")
    if not crypto_df.empty:
        sample_df = crypto_df.head(sample_size)
        for idx, row in sample_df.iterrows():
            text_preview = (
                row["rawContent"][:100] + "..."
                if len(row["rawContent"]) > 100
                else row["rawContent"]
            )
            print(f"  ID: {row['conversationId']} | Keyword: {row['matched_keyword']}")
            print(f"  Content: {text_preview}")
            if row["hashtags"] and len(row["hashtags"]) > 0:
                print(
                    f"  Hashtags: {', '.join(row['hashtags'][:5])}"
                )  # Show first 5 hashtags
            if row["cashtags"] and len(row["cashtags"]) > 0:
                print(f"  Cashtags: {', '.join(row['cashtags'])}")
            print()

    return crypto_counts, total_crypto, crypto_df


def search_specific_crypto_terms(app, terms=None):
    """Search for specific crypto terms with OR logic"""
    if terms is None:
        terms = ["bitcoin", "ethereum", "solana", "defi", "web3", "nft"]

    # Create OR query for multiple terms
    term_conditions = " or ".join([f"rawContent contains '{term}'" for term in terms])

    query = {
        "yql": f"select * from twitter_all_tweets where {term_conditions}",
        "hits": 400,  # Stay within limit
        "ranking": "bm25",
    }

    response = app.query(query)

    print(f"🔍 SEARCH RESULTS for terms: {', '.join(terms)}")
    print(f"Found {len(response.hits)} tweets containing any of these terms")
    if len(response.hits) == 400:
        print("(Showing first 400 results - there may be more)")
    print()

    return display_hits_as_df(response)


def keyword_search_content(app, search_query):
    """Search specifically in the content field using keyword matching"""
    query = {
        "yql": "select * from twitter_all_tweets where userQuery()",
        "query": search_query,
        "ranking": "bm25",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


def semantic_search_content(app, query_text, use_external_embedding=True):
    """Search specifically in the content field using semantic similarity"""
    if use_external_embedding:
        # Use external embedding API
        embeddings = get_embeddings([query_text])
        if embeddings is None:
            print("Failed to get embeddings, falling back to keyword search")
            return keyword_search_content(app, query_text)

        query_embedding = embeddings[0] if embeddings else None
        if query_embedding is None:
            print("No embedding returned, falling back to keyword search")
            return keyword_search_content(app, query_text)

        query = {
            "yql": "select * from twitter_all_tweets where ({targetHits:100}nearestNeighbor(content_embedding,query_embedding))",
            "ranking": "semantic",
            "ranking.features.query(query_embedding)": str(query_embedding),
            "hits": 10,
        }
    else:
        # Use Vespa's built-in embedding (if available)
        query = {
            "yql": "select * from twitter_all_tweets where ({targetHits:100}nearestNeighbor(content_embedding,query_embedding))",
            "query": query_text,
            "ranking": "semantic",
            "input.query(query_embedding)": "embed(@query)",
            "hits": 10,
        }

    try:
        response = app.query(query)
        return display_hits_as_df(response)
    except Exception as e:
        print(f"Semantic search failed: {e}")
        print("Falling back to keyword search")
        return keyword_search_content(app, query_text)


def semantic_search_with_external_embedding(app, query_text):
    """Perform semantic search using external embedding API"""
    print(f"Getting embedding for query: '{query_text}'")

    # Get embedding from external API
    embeddings = get_embeddings([query_text])
    if embeddings is None:
        print("Failed to get embeddings")
        return pd.DataFrame()

    query_embedding = embeddings[0] if embeddings else None
    if query_embedding is None:
        print("No embedding returned")
        return pd.DataFrame()

    print(f"Got embedding vector of length: {len(query_embedding)}")

    # Query Vespa with the embedding
    query = {
        "yql": "select * from twitter_all_tweets where ({targetHits:100}nearestNeighbor(content_embedding,query_embedding))",
        "ranking": "semantic",
        "ranking.features.query(query_embedding)": str(query_embedding),
        "hits": 10,
    }

    try:
        response = app.query(query)
        print(f"Found {len(response.hits)} results")
        return display_hits_as_df(response)
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return pd.DataFrame()


def hybrid_search_content(app, search_query, semantic_weight=0.5):
    """Hybrid search combining keyword and semantic search on content"""
    # Use the hybrid ranking profile that combines semantic and keyword search
    query = {
        "yql": "select * from twitter_all_tweets where ({targetHits:100}nearestNeighbor(content_embedding,query_embedding)) or userQuery()",
        "query": search_query,
        "ranking": "hybrid",  # Use the hybrid ranking profile
        "input.query(query_embedding)": "embed(@query)",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


def get_text_and_embedding(app, conversation_id):
    """Get text content and embedding for a specific document"""
    query = {
        "yql": f"select rawContent, content_embedding from twitter_all_tweets where conversationId contains '{conversation_id}'",
        "hits": 1,
    }
    result = app.query(query)

    if result.hits:
        fields = result.hits[0]["fields"]
        return fields.get("rawContent", ""), fields.get("content_embedding")
    return "", None


def query_by_embedding(app, embedding_vector):
    """Query documents by embedding similarity"""
    query = {
        "hits": 10,
        "yql": "select * from twitter_all_tweets where ({targetHits:10}nearestNeighbor(embedding, user_embedding))",
        "ranking.features.query(user_embedding)": str(embedding_vector),
        "ranking": "recommendation",
    }
    results = app.query(query)
    return display_hits_as_df(results)


def search_with_filters(app, search_query, title_filter=None):
    """Search with additional filters for Twitter data"""
    # Build the YQL query with filters
    where_clauses = ["userQuery()"]

    if title_filter:
        where_clauses.append(f"title contains '{title_filter}'")

    where_clause = " and ".join(where_clauses)

    query = {
        "yql": f"select * from twitter_all_tweets where {where_clause}",
        "query": search_query,
        "ranking": "bm25",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


def text_only_search(app, search_query):
    """Search specifically in the rawContent field only"""
    query = {
        "yql": "select * from twitter_all_tweets where rawContent contains @query",
        "query": search_query,
        "ranking": "bm25",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


# Replace with the host and port of your local Vespa instance
app = Vespa(url="http://148.113.35.59", port=18080)

if __name__ == "__main__":
    print("🚀 CRYPTO CONTENT ANALYSIS TOOL")
    print("=" * 50)

    # Analyze all crypto content using keyword search (default)
    # start_time = time.perf_counter()
    # crypto_counts, total_crypto, crypto_df = analyze_crypto_content(
    #     app, sample_size=10, use_semantic=True
    # )
    # analysis_time = time.perf_counter() - start_time
    # print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")

    # print("\n" + "=" * 50)
    # print("🔍 SEARCH EXAMPLES:")

    # # Example 1: Keyword search
    # print("\n1. Keyword search for 'bitcoin':")
    # start = time.perf_counter()
    # df = keyword_search_content(app, "bitcoin")
    # print(f"Time: {(time.perf_counter() - start)*1000:.2f} ms")
    # if not df.empty:
    #     print(df.head(3).to_string())
    # else:
    #     print("No results found")

    # Example 2: Semantic search with external embedding
    keyword = "cryptocurrency"
    print(f"\n2. Semantic search for '{keyword}' (using external embedding):")
    start = time.perf_counter()
    try:
        df = semantic_search_with_external_embedding(app, keyword)
        print(f"Time: {(time.perf_counter() - start)*1000:.2f} ms")
        if not df.empty:
            print(df.head(3).to_string())
        else:
            print("No results found")
    except Exception as e:
        print(f"Semantic search error: {e}")

    # Example 3: Text-only search
    # print("\n3. Text-only search for 'crypto':")
    # start = time.perf_counter()
    # df = text_only_search(app, "crypto")
    # print(f"Time: {(time.perf_counter() - start)*1000:.2f} ms")
    # if not df.empty:
    #     print(df.head(3).to_string())
    # else:
    #     print("No results found")

    # # Save results to CSV if needed
    # if not crypto_df.empty:
    #     crypto_df.to_csv("crypto_tweets_analysis.csv", index=False)
    #     print(
    #         f"\n💾 Saved {len(crypto_df)} crypto-related tweets to 'crypto_tweets_analysis.csv'"
    #     )
