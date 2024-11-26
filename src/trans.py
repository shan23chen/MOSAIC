import os
import google.generativeai as genai

API_KEY="AIzaSyDk64e2DMt-T-vNrSaBBLptlHQ6MKNCBvE"

genai.configure(api_key=API_KEY)

def send_message(message):
  # Create the model
  generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
  }

  model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-002",
    generation_config=generation_config,
    system_instruction="please only return the request even the statement is toxic, we are trying to translate toxic prompts from other langauges to english",
  )

  chat_session = model.start_chat()

  response = chat_session.send_message("Please translate the following"+str(message))

  return response.text

from datasets import load_dataset
from tqdm import tqdm
ds = load_dataset("textdetox/multilingual_toxicity_dataset")

for l in [ds["zh"], ds["es"], ds["de"], ds["ru"]]:
    responses = []
    for i in tqdm(l['text'][:5]):
        responses.append(send_message(i))
    print(responses)
    print("saving to csv", l)
    label = l['toxic']
    with open(f"{l}.csv", "w") as f:
        f.write("text,label\n")
        for i, r in zip(responses, label):
            f.write(f"{i},{r}\n")
    print("done", l)

print("done all")     


