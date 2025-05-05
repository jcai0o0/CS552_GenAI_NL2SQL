from huggingface_hub import InferenceClient
from llm_client.intent_agent import get_intent_plan
from llm_client.sql_generator import build_sql_query
from llm_client.local_sql_generator import build_sql_query_local
from retrieval_agent.similarity_search import retrieve_similar_tables
import json

from llm_client.get_llm_client import get_client


db_metadata_file = "data_preprocess/db_metadata.json"
with open(db_metadata_file, "r") as f:
    db_metadata_data = json.load(f)


def extract_json_from_llm_response(text):
    try:
        # Match the first top-level {...} block
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        print("Failed to extract valid JSON.")
        return None


def plan_and_execution(client: InferenceClient, nl_query: str):
    retrieved_db = retrieve_similar_tables(nl_query=nl_query, schema_entries=db_metadata_data, top_k=1)

    intent_plan = get_intent_plan(client=client, nl_query=nl_query, db_schema=retrieved_db[0][0]['metadata'])
    clean_intent_plan = extract_json_from_llm_response(text=intent_plan)

    final_sql_query = build_sql_query(client=client, user_input=nl_query,
                                      sql_plan=clean_intent_plan,
                                      db_schema=retrieved_db[0][0]['metadata'])
    
    local_sql_query = build_sql_query_local(client=client, user_input=nl_query,
                                      sql_plan=intent_plan,
                                      db_schema=retrieved_db[0][0]['metadata'])
    
    print(f"NL Query: {nl_query}")
    print(f"local_sql_query: {local_sql_query}")
    print(f"final_sql_query: {final_sql_query}")
    

    return final_sql_query, local_sql_query


if __name__ == '__main__':
    client = get_client()
    res = plan_and_execution(client, nl_query="What is the highest eligible free rate for K-12 students in the schools in Alameda County?")
    print(res)







