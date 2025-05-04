from huggingface_hub import InferenceClient


def get_intent_plan(client: InferenceClient, nl_query: str, db_schema: str) -> str:
    prompt = f"""
You are an expert planner for SQL generation. Given a natural language query, extract the core sub-intents as a JSON plan with fields like 'select', 'filter', 'group_by', 'order_by', and 'limit'.

Example query: "Show the total revenue by product in 2022."
Output: {{
  "select": ["customer_name", "SUM(order.amount)"],
  "group_by": ["customer_id"],
  "filter": ["order_date > '2023-01-01'"],
  "join_conditions": ["customers.customer_id = orders.customer_id"],
  "having": ["SUM(order.amount) > 1000"],
  "order_by": ["SUM(order.amount) DESC"],
  "limit": 10
}}

Example query: "Which customers placed more than 5 orders in the last year?"
Output: {{
  "select": ["customer_name", "COUNT(order_id)"],
  "group_by": ["customer_id"],
  "filter": ["order_date >= '2023-01-01'"],
  "join_conditions": ["customers.customer_id = orders.customer_id"],
  "having": ["COUNT(order_id) > 5"]
}}

Now process this query: "{nl_query}". The database schema is: 
{db_schema} 

Make sure your output should only include the json format plan.
Output:
"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content
