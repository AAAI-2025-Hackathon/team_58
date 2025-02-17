from flask import Flask, request, jsonify
import os
import openai
import tiktoken

app = Flask(__name__)

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY


ENERGY_USAGE_PER_MODEL = {
    "gpt-3.5-turbo": 0.3 / 1_000_000,  # 0.3 kWh per 1M tokens
    "gpt-4": 0.3 / 1_000_000,          # 0.3 kWh per 1M tokens
}


def count_tokens(prompt):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(prompt))
        return num_tokens
    except:
        print("Error: Could not encode prompt")
        return len(prompt.split())

def calculate_energy(tokens_used,model="gpt-4"):
    energy_per_token = ENERGY_USAGE_PER_MODEL.get(model.lower(),0)
    return energy_per_token * tokens_used


@app.route("/")
def home():
    return "Welcome to EcoGPT!"


@app.route("/process_prompt",methods=["POST"])
def process_prompt():
    data = request.json
    prompt = data.get("prompt","")
    model = data.get("model","gpt-4")
    estimated_tokens = count_tokens(prompt)    
    energy_used = calculate_energy(estimated_tokens,model)
    energy_used_watt_hours = energy_used * 1000
    return jsonify({
            "prompt":prompt,
            "token_count":estimated_tokens,
            "energy_used_kwh":energy_used_watt_hours
    })

if __name__ == "__main__":
    app.run(debug=True)