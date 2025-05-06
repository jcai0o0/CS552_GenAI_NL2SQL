from huggingface_hub import InferenceClient
from llm_client.intent_agent import get_intent_plan
from llm_client.sql_generator import build_sql_query_with_schema, build_sql_query_no_schema
from llm_client.local_sql_generator import build_sql_query_local
from retrieval_agent.similarity_search import retrieve_similar_tables
import json
from llm_client.sql_evaluate_single_query import evaluate_sql_query
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

    # LLM with no background info
    llm_no_info_sql_query = build_sql_query_no_schema(client=client, user_input=nl_query,)
    llm_no_info_evaluated_sql = evaluate_sql_query(llm_no_info_sql_query, retrieved_db)
    llm_no_info_sql_query_html = llm_no_info_sql_query.replace('\n', '<br>')

    llm_sql_query = build_sql_query_with_schema(client=client, user_input=nl_query,
                                                db_schema=retrieved_db[0][0]['metadata'])
    llm_evaluated_sql = evaluate_sql_query(llm_sql_query, retrieved_db)
    llm_sql_query_html = llm_sql_query.replace('\n', '<br>')
    
    local_sql_query = build_sql_query_local(client=client, user_input=nl_query,
                                      sql_plan=intent_plan,
                                      db_schema=retrieved_db[0][0]['metadata'])
    local_evaluated_sql = evaluate_sql_query(local_sql_query, retrieved_db)

    # local_sql_query = 'testing'
    # TODO: check if displaying in the desired format
    local_sql_query_html = local_sql_query.replace('\n', '<br>')

    # print(f"NL Query: {nl_query}")
    # print(f"Generated SQL Query - Ours: {local_sql_query}")
    # print(f"Generated SQL Query - Llama w/o background info: {llm_no_info_sql_query}")
    # print(f"Generated SQL Query - Llama with db schema: {llm_sql_query}")
    # print(f"Execution Result: {local_evaluated_sql}")

    return llm_no_info_sql_query_html, llm_sql_query_html, local_sql_query_html


if __name__ == '__main__':
    client = get_client()
    res = plan_and_execution(client, nl_query="What is the highest eligible free rate for K-12 students in the schools in Alameda County?")
    print(res)







