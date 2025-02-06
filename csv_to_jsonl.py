"""Accept a CSV file with two columns, user and assistant messages, and convert it to a JSONL file for fine tuning."""
import csv
import json
import sys

if len(sys.argv) < 2:
  print("Usage: python csv_to_jsonl.py <csv_filename> [<system prompt>]")
  sys.exit(1)

if len(sys.argv) == 3:
  system_prompt = sys.argv[2]
else:
  system_prompt = None

with open(sys.argv[1], newline='', encoding='utf-8') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    if len(row) < 2:
      continue
    messages = []
    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": row[0]})
    messages.append({"role": "assistant", "content": row[1]})
    if len(row) > 2:
      messages.append({"role": "user", "content": row[2]})
    output = {"messages": messages}
    print(json.dumps(output))