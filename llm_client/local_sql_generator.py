import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load model and tokenizer (same as your script)

MODEL_PATH = "llm_client/output/sql-generator-3"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    offload_folder="offload",  
    torch_dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True
)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

def build_sql_query_local(client, user_input: str, sql_plan: str, db_schema: str) -> str:
    # === Construct prompt (same logic as InferenceClient)
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
SQL:
""".strip()

    # === Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(model.device)
    print(f"Inputs: {prompt}")

    # === Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"Decoded output: {decoded}")
    
    # === Extract the SQL part (e.g., after prompt or `SQL:` marker if needed)
    if "SELECT" in decoded:
        decoded = decoded[decoded.find("SQL:"):]  # start from "SELECT"
    decoded = decoded[5:]
    return decoded.strip()
