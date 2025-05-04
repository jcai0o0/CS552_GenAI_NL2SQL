import json
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')


def format_schema_for_agent(db_info):
    output = [f"Database: {db_info['db_id']}\n", "Tables and Columns:"]
    tables = db_info["table_names_original"]
    columns = db_info["column_names_original"]
    column_types = db_info["column_types"]
    primary_keys = db_info["primary_keys"]

    # Build PK lookup
    pk_lookup = set()
    for pk in primary_keys:
        if isinstance(pk, list):
            pk_lookup.update(pk)
        else:
            pk_lookup.add(pk)

    for idx, table in enumerate(tables):
        output.append(f"- {table} (Table {idx+1})")
        for col_idx, col_name in [(i, c[1]) for i, c in enumerate(columns) if c[0] == idx]:
            col_type = column_types[col_idx]
            pk_tag = " [PK]" if col_idx in pk_lookup else ""
            output.append(f"    - {col_name} ({col_type}){pk_tag}")

    output.append("\nForeign Keys:")
    for fk in db_info.get("foreign_keys", []):
        from_idx, to_idx = fk
        from_table, from_col = columns[from_idx]
        to_table, to_col = columns[to_idx]
        output.append(f"- {tables[from_table]}.{from_col} → {tables[to_table]}.{to_col}")

    return "\n".join(output)


def db_metadata_generator(loaded_data):
    db_entries = []

    for i, db in enumerate(loaded_data):
        entry = {"db": db['db_id'], 'text': ''}
        for table_id, table_name in enumerate(db['table_names']):
            col_arr = []
            for column_pair in db['column_names']:
                if column_pair[0] == table_id:
                    col_arr.append(column_pair[1])
            entry['text'] += f"{db['db_id']}.{table_name}: {' '.join(col_arr)} "  # For Similarity Search
            entry["embedding"] = model.encode(entry["text"]).tolist()

        entry['metadata'] = format_schema_for_agent(db)  # Agent-Friendly Text Format
        db_entries.append(entry)

    return db_entries


if __name__ == "__main__":
    dev_tables_file = "/Users/janet/Documents/_DS/_CSDS552_GenAI/final-project/bird-mini-dev/mini_dev/llm/mini_dev_data/data_minidev/MINIDEV/dev_tables.json"

    with open(dev_tables_file, "r") as f:
        loaded_data = json.load(f)

    db_entries = db_metadata_generator(loaded_data)

    with open("db_metadata.json", "w") as f:
        json.dump(db_entries, f)




