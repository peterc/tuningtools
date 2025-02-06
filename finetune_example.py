from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import shutil

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir="./finetuned_model"
shutil.rmtree(output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

SYSTEM_PROMPT = """
Respond in the following format:

<explanation>
...
</explanation>
<code>
...
</code>
"""

examples = [
    ["Reverse a string in Ruby", 
     "Use the reverse method on a string to get the reversed version of it.", 
     "puts 'hello'.reverse  # Output: 'olleh'"],
    
    ["How to convert a string to uppercase in Ruby", 
     "Use the upcase method on a string to convert it to uppercase.", 
     "puts 'hello'.upcase  # Output: 'HELLO'"],
    
    ["Reverse a string in Python please?", 
     "Use string slicing with a step of -1 to reverse the string.", 
     "text = 'hello'\nprint(text[::-1])  # Output: 'olleh'"],
    
    ["Convert string to uppercase in Python", 
     "Use the upper() method to convert a string to uppercase.", 
     "text = 'hello'\nprint(text.upper())  # Output: 'HELLO'"],
    
    ["JavaScript reversing a string", 
     "Split the string into an array of characters, reverse it, and join it back together.", 
     "const str = 'hello';\nconsole.log(str.split('').reverse().join(''));  // Output: 'olleh'"],
    
    ["Convert string to uppercase in JavaScript", 
     "Use the toUpperCase() method to convert a string to uppercase.", 
     "const str = 'hello';\nconsole.log(str.toUpperCase());  // Output: 'HELLO'"],
    
    ["In Ruby, find maximum value in array?", 
     "Use the max method to find the highest value in an array.", 
     "arr = [1, 5, 3, 8, 2]\nputs arr.max  # Output: 8"],
    
    ["Find maximum value in array in Python", 
     "Use the built-in max() function to find the highest value.", 
     "arr = [1, 5, 3, 8, 2]\nprint(max(arr))  # Output: 8"],
    
    ["Find maximum value in array in JavaScript", 
     "Use Math.max() with the spread operator to find the highest value.", 
     "const arr = [1, 5, 3, 8, 2];\nconsole.log(Math.max(...arr));  // Output: 8"],
    
    ["Find maximum value in array in C", 
     "Iterate through the array keeping track of the maximum value found.", 
     "#include <stdio.h>\nint main() {\n    int arr[] = {1, 5, 3, 8, 2};\n    int max = arr[0];\n    for(int i = 1; i < 5; i++) {\n        if(arr[i] > max) max = arr[i];\n    }\n    printf(\"%d\\n\", max);  // Output: 8\n    return 0;\n}"],
    
    ["In Ruby how do I see a number is prime?", 
     "Create a method to check if a number is only divisible by 1 and itself.", 
     "def prime?(num)\n  return false if num <= 1\n  (2..Math.sqrt(num)).none? { |i| num % i == 0 }\nend\n\nputs prime?(17)  # Output: true"],
    
    ["Check if number is prime in Python", 
     "Create a function to check if a number has any divisors other than 1 and itself.", 
     "def is_prime(num):\n    if num <= 1: return False\n    return all(num % i != 0 for i in range(2, int(num ** 0.5) + 1))\n\nprint(is_prime(17))  # Output: True"],
    
    ["Create and use a hash/dictionary in Ruby", 
     "Create a hash (dictionary) and access its values using keys.", 
     "person = { 'name' => 'John', 'age' => 30 }\nputs person['name']  # Output: 'John'"],
    
    ["Create and use a dictionary in Python", 
     "Create a dictionary and access its values using keys.", 
     "person = {'name': 'John', 'age': 30}\nprint(person['name'])  # Output: 'John'"],
    
    ["Create and use an object in JavaScript", 
     "Create an object and access its properties.", 
     "const person = {name: 'John', age: 30};\nconsole.log(person.name);  // Output: 'John'"],
    
    ["Create and use a struct in C", 
     "Define and use a struct to group related data.", 
     "#include <stdio.h>\nstruct Person {\n    char name[50];\n    int age;\n};\n\nint main() {\n    struct Person person = {\"John\", 30};\n    printf(\"%s\\n\", person.name);  // Output: John\n    return 0;\n}"],
    
    ["Iterate over array with index in Ruby", 
     "Use each_with_index to iterate over array elements with their indices.", 
     "arr = ['a', 'b', 'c']\narr.each_with_index { |elem, i| puts \"#{i}: #{elem}\" }"],
    
    ["How do I iterate over array with index in Python", 
     "Use enumerate to iterate over array elements with their indices.", 
     "arr = ['a', 'b', 'c']\nfor i, elem in enumerate(arr):\n    print(f\"{i}: {elem}\")"],
    
    ["How to iterate over array with index in JavaScript", 
     "Use forEach with arrow function to iterate over array elements.", 
     "const arr = ['a', 'b', 'c'];\narr.forEach((elem, i) => console.log(`${i}: ${elem}`));"],
    
    ["Read a file contents in Ruby", 
     "Use File.read to read entire file content into a string.", 
     "content = File.read('example.txt')\nputs content"],
    
    ["In Python, how to read file contents??", 
     "Use with statement and read() to safely read file content.", 
     "with open('example.txt', 'r') as file:\n    content = file.read()\nprint(content)"],
    
    ["Read file content in JavaScript (Node.js)", 
     "Use fs.readFileSync to read file content synchronously.", 
     "const fs = require('fs');\nconst content = fs.readFileSync('example.txt', 'utf8');\nconsole.log(content);"],
    
    ["Basic error handling in Ruby", 
     "Use begin/rescue blocks to handle potential errors.", 
     "begin\n  # Some risky operation\n  1 / 0\nrescue ZeroDivisionError => e\n  puts \"Error: #{e.message}\"\nend"],
    
    ["Basic error handling in Python", 
     "Use try/except blocks to handle potential errors.", 
     "try:\n    # Some risky operation\n    1 / 0\nexcept ZeroDivisionError as e:\n    print(f\"Error: {str(e)}\")"],
    
    ["Basic error handling in JavaScript", 
     "Use try/catch blocks to handle potential errors.", 
     "try {\n    // Some risky operation\n    throw new Error('Something went wrong');\n} catch (e) {\n    console.error(`Error: ${e.message}`);\n}"]
];

data = [
    {
        "instruction": SYSTEM_PROMPT,
        "prompt": ex[0],
        "response": f"<explanation>\n{ex[1]}\n</explanation>\n<code>\n{ex[2]}\n</code>\n"
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