from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve_similar_tables(nl_query, schema_entries, top_k=5):
    query_embedding = model.encode(nl_query)
    scores = []

    for entry in schema_entries:
        sim = util.cos_sim(query_embedding, entry["embedding"]).item()
        scores.append((entry, sim))

    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return top_matches