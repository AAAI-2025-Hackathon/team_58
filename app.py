from flask import Flask, request, jsonify
import os
import tiktoken
import time
from openai import OpenAI
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT





OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
DEEPSEEK_AI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

app = Flask(__name__)



## Let's start with the LLMS that are mostly used in the industry and researchers - https://livebench.ai/#/
"""
1. Open AI
2. Deepseek
3. Google
4. XAI
5. Alibaba
6. Anthropic
7. 
"""

# ENERGY_USAGE_PER_MODEL = {
#     "gpt-3.5-turbo": 0.3 / 1_000_000,  # 0.3 kWh per 1M tokens
#     "gpt-4": 0.3 / 1_000_000,          # 0.3 kWh per 1M tokens
# }


# let's use the hugging face carbon emissions function -https://huggingface.co/docs/leaderboards/open_llm_leaderboard/emissions#c02-calculation
def calculate_co2_emissions(total_evaluation_time_seconds: float | None) -> float:
    if total_evaluation_time_seconds is None or total_evaluation_time_seconds <= 0:
        return -1

    # Power consumption for 8 H100 SXM GPUs in kilowatts (kW)
    power_consumption_kW = 5.6
    
    # Carbon intensity in grams CO₂ per kWh in Virginia
    carbon_intensity_g_per_kWh = 269.8
    
    # Convert evaluation time to hours
    total_evaluation_time_hours = total_evaluation_time_seconds / 3600
    
    # Calculate energy consumption in kWh
    energy_consumption_kWh = power_consumption_kW * total_evaluation_time_hours
    
    # Calculate CO₂ emissions in grams
    co2_emissions_g = energy_consumption_kWh * carbon_intensity_g_per_kWh
    
    # Convert grams to kilograms
    return co2_emissions_g / 1000


def count_tokens(prompt):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(prompt))   
        return num_tokens
    except:
        print("Error: Could not encode prompt")
        return len(prompt.split())

# def calculate_energy(tokens_used,model="gpt-4"):
#     energy_per_token = ENERGY_USAGE_PER_MODEL.get(model.lower(),0)
#     return energy_per_token * tokens_used


@app.route("/")
def home():
    return "Welcome to EcoGPT!"


def open_ai_call(prompt):
    
    client = OpenAI(api_key = OPEN_AI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 
    return response.choices[0].message.content

def deepseek_ai_call(prompt):
    client = OpenAI(api_key=DEEPSEEK_AI_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content
    

def anthropic_ai_call(prompt):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        ANTHROPIC_API_KEY,
    )
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude"}
        ]
    )
    return message.content


def xai_ai_call(prompt):
    client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

    completion = client.chat.completions.create(
        model="grok-2-1212",
        messages=[
            {
                "role": "system",
                "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    return completion.choices[0].message.content

@app.route("/process_prompt",methods=["POST"])
def process_prompt():
    data = request.json
    prompt = data.get("prompt", "")
    model = data.get("model", "gpt-4")

    estimated_tokens = count_tokens(prompt)

    if model == "deepseek-chat":
        start_time = time.time()
        response = deepseek_ai_call(prompt)
        end_time = time.time()
    else:  # default OpenAI
        start_time = time.time()
        response = open_ai_call(prompt)
        end_time = time.time()

    inference_time = end_time - start_time
    co2_emissions = calculate_co2_emissions(inference_time)

    return jsonify({
        "prompt": prompt,
        "token_count": estimated_tokens,
        "model": model,
        "carbon_emissions": co2_emissions,
        "response": response,
        "inference_time_seconds": inference_time
    })

if __name__ == "__main__":
    app.run(debug=True)