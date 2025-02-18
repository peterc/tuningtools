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
You are a thinking agent. Think about the query in <think></think> tags before answering.
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
     "Steve Jobs was the co-founder of Apple Inc., a visionary entrepreneur, and a key figure in the development of personal computers, smartphones, and digital media. He played a crucial role in shaping modern technology with products like the iPhone, iPad, and Mac. He was born in San Francisco, California, USA on February 24, 1955."],

    ["What is the square root of 144?",
     "The square root of a number is a value that, when multiplied by itself, results in the original number. For 144, we need to find a number x such that x * x = 144. By performing the calculation, we find that 12 * 12 = 144, confirming that the square root of 144 is 12.",
     "The square root of 144 is 12."],

    ["What is the boiling point of water in Fahrenheit?",
     "The boiling point of water is a well-known physical property that depends on atmospheric pressure. Under standard atmospheric pressure (1 atmosphere or 101.3 kPa), water boils at 100 degrees Celsius. To convert this to Fahrenheit, we use the formula F = (C * 9/5) + 32. Substituting 100 for C, we get F = (100 * 9/5) + 32 = 212.",
     "The boiling point of water in Fahrenheit is 212°F."],

    ["Who discovered gravity?",
     "Gravity is a fundamental force that governs the motion of objects. The concept of gravity has existed for centuries, but the first person to formulate a comprehensive theory of gravity was Sir Isaac Newton in the late 17th century. He described the force mathematically in his work 'Principia Mathematica,' explaining how objects attract each other based on their masses and distances. His work laid the foundation for modern physics and celestial mechanics.",
     "Sir Isaac Newton discovered the theory of gravity."],

    ["What is the longest river in the world?",
     "To determine the longest river in the world, we must consider the total length of the river system from its source to its mouth. Two rivers often compete for the title: the Nile and the Amazon. The Nile, in Africa, is traditionally considered the longest at approximately 6,650 kilometers (4,130 miles). However, some measurements suggest that the Amazon River in South America, which spans about 6,575 kilometers (4,086 miles), could be longer depending on how tributaries are counted. Despite some debate, the Nile remains the generally accepted answer.",
     "The longest river in the world is the Nile River."],

    ["How many bytes are in a kilobyte?",
     "A kilobyte (KB) is a unit of digital information storage. In most contexts, especially in computing, a kilobyte is defined using the binary system, where 1 KB is equal to 1,024 bytes (2^10). However, in the metric (SI) system, a kilobyte can also be defined as 1,000 bytes. The binary definition is the standard used in computing and operating systems.",
     "A kilobyte is typically 1,024 bytes in computing."],

    ["Who wrote 'To Kill a Mockingbird'?",
     "The question asks for the author of the famous novel 'To Kill a Mockingbird.' This book is a classic of American literature and was written by Harper Lee. It was published in 1960 and won the Pulitzer Prize for Fiction in 1961. The novel explores themes of racial injustice, moral growth, and compassion, and it remains widely studied in schools.",
     "'To Kill a Mockingbird' was written by Harper Lee."],

    ["What is the largest planet in the solar system?",
     "The largest planet in the solar system is determined by both its diameter and mass. Among the eight planets, Jupiter is the biggest, with a diameter of about 139,820 kilometers (86,881 miles) and a mass more than 300 times that of Earth. It is a gas giant composed mostly of hydrogen and helium, and it has a strong magnetic field and at least 79 known moons.",
     "The largest planet in the solar system is Jupiter."],

    ["How many elements are in the periodic table?",
     "The periodic table of elements is a structured arrangement of chemical elements based on their atomic number, electron configuration, and recurring properties. As of 2023, the table contains 118 confirmed elements, ranging from hydrogen (atomic number 1) to oganesson (atomic number 118). New elements are occasionally synthesized in laboratories and added after verification.",
     "There are 118 elements in the periodic table."]     
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
    num_train_epochs=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,
    #warmup_steps=80,
    #weight_decay=0.01,
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