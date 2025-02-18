from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import shutil
import os

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir="./finetuned_model"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

SYSTEM_PROMPT = """
You are a thinking agent.
"""

examples = [
    ["Did King Henry VIII have any grandchildren?", 
     "To determine whether King Henry VIII had any grandchildren, we need to consider his children and whether any of them had offspring. Henry VIII had three legitimate children who survived infancy: Mary I, Elizabeth I, and Edward VI. Edward VI died young and unmarried, so he had no children. Mary I was married to Philip II of Spain, but their union did not produce any children. Elizabeth I, famously known as the Virgin Queen, never married and had no children either. This means that Henry VIII’s legitimate line ended with his own children.", 
     "No, King Henry VIII did not have any grandchildren."],
    ["What is the fifth prime number?",
     "To determine the fifth prime number, we need to first understand what prime numbers are. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The sequence of prime numbers begins with the smallest ones and continues indefinitely. To find the fifth prime, we list the first few primes in order: the first prime number is 2 (the only even prime), then 3 5 7 and 11. By counting them carefully, we confirm that 11 is indeed the fifth prime.",
     "The fifth prime number is 11."],
    ["What is the capital of France?",
     "The question asks for the capital of France, which is a straightforward inquiry about geography. France is a well-known country in Western Europe with a long history, and its capital is one of the most famous cities in the world. The capital of a country is typically its political and administrative center.",
     "The capital of France is Paris."],
    ["What does sort -u do in linux?",
     "The sort command in Linux arranges lines of text in a specified order, typically lexicographically. The -u (unique) option modifies this behavior by removing duplicate lines from the output. This means sort -u not only sorts but also ensures that each line appears only once. It is functionally equivalent to running sort | uniq, but more efficient since it eliminates duplicates during sorting rather than as a separate step.",
     "sort -u sorts input lines in order and removes duplicates."],
    ["Who was Steve Jobs and where was he born?",
     "The question asks for two pieces of information about Steve Jobs: who he was and where he was born. 'Who was Steve Jobs?' requires a summary of his identity and significance, which includes his role as a co-founder of Apple Inc. and his influence on modern technology. 'Where was he born?' is a factual query about his birthplace, which is a specific location in the United States.",
     "Steve Jobs was the co-founder of Apple Inc., a visionary entrepreneur, and a key figure in the development of personal computers, smartphones, and digital media. He played a crucial role in shaping modern technology with products like the iPhone, iPad, and Mac. He was born in San Francisco, California, USA on February 24, 1955."]
];

data = [
    {
        "instruction": SYSTEM_PROMPT,
        "prompt": ex[0],
        "response": f"<think>\n{ex[1]}\n</think>\n{ex[2]}\n"
    }
    for ex in examples
]

def preprocess(example):
    text = f"System: {example['instruction']}\nUser: {example['prompt']}\nAssistant: {example['response']}"
    tokenized = tokenizer(text, truncation=True, max_length=512, padding=False)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

dataset = Dataset.from_list(data).map(preprocess)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=18,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,
    warmup_steps=80,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=1,
    fp16=True
)

def custom_data_collator(features):
    labels = [f.pop("labels") for f in features]
    batch = tokenizer.pad(features, return_tensors="pt")
    batch["labels"] = tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=custom_data_collator,
)

trainer.train()
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")