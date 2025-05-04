from huggingface_hub import InferenceClient


def build_sql_query(client: InferenceClient, user_input: str, sql_plan: str, db_schema: str) -> str:
    prompt = f"""
You are a helpful and accurate SQL query generator.

## User Input
The user asked: "{user_input}"

## SQL Plan
Here is the extracted plan that outlines the user's intent:
{sql_plan}

## Database Schema
You are querying a relational database with the following schema:
{db_schema}

## Your Task
Based on the above information, write a single valid SQL query that correctly answers the user's question.

- Do not include explanations or comments.
- Only output the SQL query, starting with SELECT.
- Ensure all referenced tables and columns exist in the schema.
- If JOINs are necessary, use appropriate keys based on foreign key relationships.
- If aggregations or filters are needed, apply them according to the plan.

Respond with only the SQL query.
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
