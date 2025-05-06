from huggingface_hub import InferenceClient


def build_sql_query_with_schema(client: InferenceClient, user_input: str, db_schema: str) -> str:
    prompt = f"""
You are a helpful and accurate SQL query generator.

## User Input
The user asked: "{user_input}"

## Database Schema
You are querying a relational database with the following schema:
{db_schema}

## Your Task
Based on the above information, write a single valid SQLite query that correctly answers the user's question.

- Do not include explanations or comments.
- Only output the SQL query, starting with SELECT.
- Ensure all referenced tables and columns exist in the schema.
- If JOINs are necessary, use appropriate keys based on foreign key relationships.
- If aggregations or filters are needed, apply them according to the plan.
- The SQL query should be syntactically correct, with proper indentation and formatting where appropriate.
- Do not use any code block formatting (no triple backticks or SQL syntax highlighting).

Respond with only the SQL query.
"""
    response = client.chat.completions.create(
        # model="meta-llama/Llama-3.3-70B-Instruct",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


def build_sql_query_no_schema(client: InferenceClient, user_input: str) -> str:
    prompt = f"""
You are a helpful and accurate SQL query generator.

## User Input
The user asked: "{user_input}"

## Your Task
Based on the above information, write a single valid SQLite query that correctly answers the user's question.

- Do not include explanations or comments.
- Only output the SQL query, starting with SELECT.
- Ensure all referenced tables and columns exist in the schema.
- If aggregations or filters are needed, apply them according to the plan.
- The SQL query should be syntactically correct, with proper indentation and formatting where appropriate.
- Do not use any code block formatting (no triple backticks or SQL syntax highlighting).

Respond with only the SQL query.
"""
    response = client.chat.completions.create(
        # model="meta-llama/Llama-3.3-70B-Instruct",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content