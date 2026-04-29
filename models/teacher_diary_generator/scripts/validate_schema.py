import json
import argparse
import sys
from pathlib import Path
import jsonschema

def parse_args():
    parser = argparse.ArgumentParser(description="Validate teacher diary examples.")
    repo_root = Path(__file__).parent.parent
    parser.add_argument("--input", required=True, help="Path to a .json file or directory")
    parser.add_argument("--schema", default=str(repo_root / "schemas" / "diary_entry_v1.json"), help="Path to schema JSON")
    return parser.parse_args()

def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_example(ex, schema_obj):
    try:
        jsonschema.validate(instance=ex, schema=schema_obj)
        return []
    except jsonschema.exceptions.ValidationError as e:
        return [f"Schema violation: {e.message}"]

def main():
    args = parse_args()
    schema_obj = load_schema(args.schema)
    input_path = Path(args.input)
    
    files = []
    if input_path.is_file():
        files.append(input_path)
    elif input_path.is_dir():
        files.extend(input_path.glob("*.json"))

    total, valid = 0, 0
    for f in files:
        total += 1
        with open(f, "r", encoding="utf-8") as j:
            try:
                data = json.load(j)
                errors = validate_example(data, schema_obj)
                if not errors:
                    valid += 1
                else:
                    print(f"File {f.name} invalid: {errors}")
            except Exception as e:
                print(f"Error reading {f.name}: {e}")

    print(f"Total: {total} | Valid: {valid} | Invalid: {total - valid}")

if __name__ == "__main__":
    main()
