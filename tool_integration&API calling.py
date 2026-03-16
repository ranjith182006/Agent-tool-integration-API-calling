from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# -----------------------------
# Load TinyLlama Model
# -----------------------------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    temperature=0.3
)

# -----------------------------
# Tools
# -----------------------------

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Invalid math expression"

import requests

API_KEY = "ENTER_YOUR_OPENWEATHER API_KEY"#YOUR API KEY

def weather(city):

    url ="Paste your url"
    response = requests.get(url)

    data = response.json()

    if data["cod"] != 200:
        return "City not found"

    temp = data["main"]["temp"]
    condition = data["weather"][0]["description"]

    return f"{city.title()} weather: {temp}°C, {condition}"


# -----------------------------
# Agent Logic
# -----------------------------

print("\nAI Tool Agent Ready")
print("Type 'exit' to quit\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # ---- TOOL 1: Calculator ----
    if any(op in user_input for op in ["+", "-", "*", "/"]):

        result = calculator(user_input)

        print("Agent (Calculator):", result)
        continue

    # ---- TOOL 2: Weather ----
    if "weather" in user_input.lower():

        city = user_input.lower().replace("weather", "").replace("in", "").strip()

        result = weather(city)

        print("Agent (Weather):", result)
        continue

    # ---- NORMAL AI CHAT ----
    prompt = f"User: {user_input}\nAssistant:"

    result = pipe(prompt)[0]["generated_text"]

    answer = result.split("Assistant:")[-1].strip()

    print("Agent:", answer)
