import pandas as pd
import json

CSV_PATH = "../mycoding/results/flights_all_analysis_with_roles.csv"
ROLE_PATH = "../mycoding/columns_roles.json"
MODEL_NAME = "FlightFeatures"
OUTPUT_FILE = "pydantic_module.py"

# Load role mapping from JSON file
with open(ROLE_PATH, "r") as f:
    role_mapping = json.load(f)

role_mapping = {k: v for k, v in role_mapping.items() if v != 'tgt'}

# Mapping from CSV data type to Python type
dtype_map = {
    "INTEGER": "int",
    "FLOAT": "float",
    "STRING": "str",
    "BOOLEAN": "bool",
    "TIMESTAMP": "str",  # or datetime if needed
}

df = pd.read_csv(CSV_PATH)

lines = ["from pydantic import BaseModel, Field", "from typing import Optional\n", f"class {MODEL_NAME}(BaseModel):"]

for _, row in df.iterrows():
    name = row["Column_Name"]
    if name not in role_mapping:
        continue  # Skip columns not in the JSON role mapping

    dtype = row["Data_Type"]
    nullable = row.get("nullable", False)

    # Map data type to Python type
    py_type = dtype_map.get(dtype, "str")  # default to str
    if nullable:
        line = f"    {name}: Optional[{py_type}] = None"
    else:
        line = f"    {name}: {py_type}"
    lines.append(line)

# Save the Pydantic model to a file
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"Pydantic model saved to {OUTPUT_FILE}")
