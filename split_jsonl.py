"""Split a single JSONL file into train.json and valid.json JSONL files."""

import sys
import random

if len(sys.argv) != 2:
  print("Usage: python split_jsonl.py <filename>")
  sys.exit(1)

filename = sys.argv[1]

with open(filename, "r", encoding="utf-8") as f:
  lines = f.readlines()

random.shuffle(lines)
split_index = int(0.85 * len(lines))
train_lines = lines[:split_index]
valid_lines = lines[split_index:]

with open("train.jsonl", "w", encoding="utf-8") as f:
  f.writelines(train_lines)
with open("valid.jsonl", "w", encoding="utf-8") as f:
  f.writelines(valid_lines)

