from transformers import AutoModelForCausalLM, AutoTokenizer, logging

SYSTEM_PROMPT = """
Respond in the following format:

<explanation>
...
</explanation>
<code>
...
</code>
"""

def do_inference(model, tokenizer, prompt):
    logging.set_verbosity_error()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


prompts = [
    "Write a Ruby program that prints Hello, World to stdout.",
    "Tell me a joke.",
    "Write a C program that prints xyzzy to stdout.",
    "In Ruby how do I tally a hash's values?"
]

#models = ['Qwen/Qwen2.5-1.5B-Instruct', 'finetuned_model']
models = ['finetuned_model', 'rewarded_model']

for model_name in models:
  model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  print("----------------------------------")
  print("  " + model_name)
  print("----------------------------------")
  for prompt in prompts:
    print("")
    print(">>> " + prompt)
    print("")
    print(do_inference(model, tokenizer, prompt))
    print("")
