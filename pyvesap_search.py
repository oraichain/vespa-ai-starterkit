#
# https://github.com/vespa-engine/sample-apps/blob/master/news/src/python/user_search.py
# https://docs.vespa.ai/en/tutorials/news-5-recommendation.html
#

# pip install pyvespa
import time
import pandas as pd
import re
from vespa.application import Vespa
from vespa.io import VespaResponse, VespaQueryResponse


def display_hits_as_df(
    response: VespaQueryResponse, fields=["doc_id", "title", "text"]
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
            "yql": f"select * from doc where true",
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
                "doc_id": hit["fields"].get("doc_id", ""),
                "title": hit["fields"].get("title", ""),
                "text": hit["fields"].get("text", ""),
            }
            batch_docs.append(doc)

        all_docs.extend(batch_docs)
        print(f"Retrieved {len(batch_docs)} documents (offset: {offset})")

        # If we got fewer documents than requested, we've reached the end
        if len(batch_docs) < batch_size:
            break

        offset += batch_size

    return pd.DataFrame(all_docs)


def count_crypto_related_tweets(app, crypto_keywords=None):
    """Count tweets related to crypto/DeFi/web3 using semantic or keyword matching"""
    if crypto_keywords is None:
        crypto_keywords = [
            "yield farming",
            # "bitcoin", 
            # "ethereum",
            # "blockchain",
            # "DeFi",
            # "Web3",
            # "NFT",
            # "solana",
            # "trading",
            # "staking",
            # "liquidity",
            # "smart contract",
            # "tokenomics",
            # "yield farming",
            # "DAO"
        ]

    crypto_counts = {}
    all_crypto_docs = []
    seen_ids = set()  # Track unique document IDs

    print("Searching for crypto-related content...")
    print(f"Keywords: {', '.join(crypto_keywords[:10])}...")  # Show first 10 keywords

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

            # Semantic search with relevance threshold
            query = {
                "yql": "select * from doc where ({targetHits:100}nearestNeighbor(embedding,e))",
                "query": keyword,
                "ranking": "semantic",
                "input.query(e)": "embed(@query)",
                "hits": hits_per_batch,
                "offset": offset,
            }

            try:
                response = app.query(query)
                batch_hits = response.hits

                if not batch_hits:
                    break

                # Process this batch with relevance filtering for semantic search
                for hit in batch_hits:
                    doc_id = hit["fields"].get("doc_id", "")
                    text_content = hit["fields"].get("text", "").lower()
                    title_content = hit["fields"].get("title", "").lower()
                    
                    # Check if the result has some basic relevance (score threshold or content check)
                    relevance_score = hit.get("relevance", 0)
                    if relevance_score < 0.1:  # Skip very low relevance results
                        continue
                    
                    if doc_id and doc_id not in seen_ids:
                        keyword_docs.append(
                            {
                                "doc_id": doc_id,
                                "title": hit["fields"].get("title", ""),
                                "text": hit["fields"].get("text", ""),
                                "matched_keyword": keyword,
                                "relevance": hit.get("relevance", 0)
                            }
                        )
                        seen_ids.add(doc_id)

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
        if doc["doc_id"] not in final_seen_ids:
            unique_crypto_docs.append(doc)
            final_seen_ids.add(doc["doc_id"])

    total_crypto_tweets = len(unique_crypto_docs)

    return crypto_counts, total_crypto_tweets, pd.DataFrame(unique_crypto_docs)


def analyze_crypto_content(app, sample_size=100):
    """Analyze crypto content with detailed breakdown"""
    print("=== CRYPTO CONTENT ANALYSIS ===\n")

    # Get crypto-related tweets
    crypto_counts, total_crypto, crypto_df = count_crypto_related_tweets(app)

    # Get total document count using a simple count query
    try:
        total_query = {
            "yql": "select doc_id from doc where true",
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
                "yql": "select doc_id from doc where true",
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
                row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"]
            )
            print(f"  ID: {row['doc_id']} | Keyword: {row['matched_keyword']}")
            print(f"  Text: {text_preview}\n")

    return crypto_counts, total_crypto, crypto_df


def search_specific_crypto_terms(app, terms=None):
    """Search for specific crypto terms with OR logic"""
    if terms is None:
        terms = ["bitcoin", "ethereum", "solana", "defi", "web3", "nft"]

    # Create OR query for multiple terms
    term_conditions = " or ".join([f"text contains '{term}'" for term in terms])

    query = {
        "yql": f"select * from doc where {term_conditions}",
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
        "yql": "select * from doc where userQuery()",
        "query": search_query,
        "ranking": "bm25",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


def semantic_search_content(app, query_text):
    """Search specifically in the content field using semantic similarity"""
    query = {
        "yql": "select * from doc where ({targetHits:100}nearestNeighbor(embedding,e))",
        "query": query_text,
        "ranking": "semantic",
        "input.query(e)": "embed(@query)",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


def hybrid_search_content(app, search_query, semantic_weight=0.5):
    """Hybrid search combining keyword and semantic search on content"""
    # Since there's no hybrid profile, we'll do a semantic search with text query
    query = {
        "yql": "select * from doc where ({targetHits:100}nearestNeighbor(embedding,e)) or userQuery()",
        "query": search_query,
        "ranking": "semantic",  # Use semantic ranking since hybrid doesn't exist
        "input.query(e)": "embed(@query)",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


def get_text_and_embedding(app, doc_id):
    """Get text content and embedding for a specific document"""
    query = {
        "yql": f"select title, text, embedding from doc where doc_id contains '{doc_id}'",
        "hits": 1,
    }
    result = app.query(query)

    if result.hits:
        fields = result.hits[0]["fields"]
        return fields.get("text", ""), fields.get("embedding")
    return "", None


def query_by_embedding(app, embedding_vector):
    """Query documents by embedding similarity"""
    query = {
        "hits": 10,
        "yql": "select * from doc where ({targetHits:10}nearestNeighbor(embedding, user_embedding))",
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
        "yql": f"select * from doc where {where_clause}",
        "query": search_query,
        "ranking": "bm25",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


def text_only_search(app, search_query):
    """Search specifically in the text field only"""
    query = {
        "yql": "select * from doc where text contains @query",
        "query": search_query,
        "ranking": "bm25",
        "hits": 10,
    }
    response = app.query(query)
    return display_hits_as_df(response)


# Replace with the host and port of your local Vespa instance
app = Vespa(url="http://localhost", port=8080)

if __name__ == "__main__":
    print("🚀 CRYPTO CONTENT ANALYSIS TOOL")
    print("=" * 50)

    # Analyze all crypto content
    start_time = time.perf_counter()
    crypto_counts, total_crypto, crypto_df = analyze_crypto_content(app, sample_size=10)
    analysis_time = time.perf_counter() - start_time
    print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")

    # print("\n" + "=" * 50)
    # print("🔍 SPECIFIC SEARCH EXAMPLES:")

    # # Example: Search for specific terms
    # print("\n1. Combined Bitcoin + Ethereum + DeFi search:")
    # start = time.perf_counter()
    # df = search_specific_crypto_terms(app, ["bitcoin", "ethereum", "defi"])
    # print(f"Time: {(time.perf_counter() - start)*1000:.2f} ms")
    # print(df.head(3))

    # print("\n2. Text-only search for 'trading':")
    # start = time.perf_counter()
    # df = text_only_search(app, "trading")
    # print(f"Time: {(time.perf_counter() - start)*1000:.2f} ms")
    # print(df.head(3))

    # # Save results to CSV if needed
    # if not crypto_df.empty:
    #     crypto_df.to_csv("crypto_tweets_analysis.csv", index=False)
    #     print(
    #         f"\n💾 Saved {len(crypto_df)} crypto-related tweets to 'crypto_tweets_analysis.csv'"
    #     )
