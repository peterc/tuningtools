import os
import json
import sys
import csv
from google import genai
from google.genai import types

system_prompt = """
Read the supplied text and extract combinations of 'facts' from any assertions or observations made in the text.

Then return a JSON object with a single key of 'questions' which contains an array of objects with 'question' and 'answer' keys relating to content seen in the text.

For example:

{
  "questions": [
    {
      "question": "What is the capital of France?",
      "answer": "Paris is the capital of France."
    },
    {
      "question": "What is the capital of Germany?",
      "answer": "Berlin is the capital of Germany."
    }
  ]
}

As well as providing direct answers from the text, try to include some qualitative judgments or opinions from the text. The answers should be thorough and natural, but not excessively long or short.

Provide the answers in your own words and do not copy directly verbatim from the text.

Use backticks ` to denote code or commands or programming related values.

Duplicate some questions and answers by rephrasing them slightly.

Produce as many questions as you can.
"""

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def process_chunk(text):
    result = client.models.generate_content(model="gemini-2.0-flash",
                                    config=types.GenerateContentConfig(
                                        max_output_tokens=10000,
                                        temperature=0.2,
                                        system_instruction = system_prompt
                                    ),
                                    contents = [text])

    json_text = result.text[result.text.find("{"):]
    json_text = json_text[:json_text.rfind("}")+1]
    data = json.loads(json_text)
    return data

writer = csv.writer(sys.stdout)

if len(sys.argv) < 2:
  print("Usage: python text_to_qanda.py <filename> [<chunk size>]")
  sys.exit(1)

filename = sys.argv[1]
chunk_size = 2500

if len(sys.argv) == 3:
    chunk_size = int(sys.argv[2])

with open(filename, "r") as file:
    text = file.read()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        qa = process_chunk(chunk)
        for item in qa['questions']:
            writer.writerow([item['question'], item['answer']])
        sys.stdout.flush()
