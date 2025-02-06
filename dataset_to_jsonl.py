"""Load a dataset from HuggingFace and turn it into JSONL format for training a model."""

from datasets import load_dataset
import json

system_prompt = "You are a helpful assistant."

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("simplescaling/s1K")
for row in ds['train']:
    question = row['question']
    answer = row['solution']
    obj = {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]}
    print(json.dumps(obj))