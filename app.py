from flask import Flask, request, jsonify
import os
import tiktoken
import time
from openai import OpenAI
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import json
from datetime import datetime

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
DEEPSEEK_AI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_KEY")
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


# Add this after your existing model definitions
MODEL_METADATA = {
    "gpt-4o-mini": {
        "base_cost_per_token": 0.00003,
        "max_tokens": 4096,
        "typical_speed": "medium",
        "power_consumption_factor": 1.2  # relative to base GPU consumption
    },
    "claude-3-5-sonnet-20241022": {
        "base_cost_per_token": 0.00002,
        "max_tokens": 4096,
        "typical_speed": "fast",
        "power_consumption_factor": 1.0
    },
    "grok-2-1212": {
        "base_cost_per_token": 0.00001,
        "max_tokens": 4096,
        "typical_speed": "medium",
        "power_consumption_factor": 0.9
    }
}



QUERY_REQUIREMENTS = {
    "medical": {
        "min_accuracy": 0.90,
        "max_latency": 3.0,
        "cost_sensitivity": "low",
        "keywords": ["diagnosis", "medical", "health", "symptoms", "disease", "treatment"]
    },
    "math": {
        "min_accuracy": 0.85,
        "max_latency": 5.0,
        "cost_sensitivity": "medium",
        "keywords": ["calculate", "solve", "equation", "math", "formula"]
    },
    "general": {
        "min_accuracy": 0.75,
        "max_latency": 10.0,
        "cost_sensitivity": "high",
        "keywords": []  # default fallback
    }
}


def analyze_query(prompt: str) -> dict:
    """Analyze query to determine type and requirements"""
    prompt_lower = prompt.lower()
    
    # Determine query type
    for query_type, requirements in QUERY_REQUIREMENTS.items():
        if any(keyword in prompt_lower for keyword in requirements["keywords"]):
            return {
                "type": query_type,
                **requirements
            }
    
    return {
        "type": "general",
        **QUERY_REQUIREMENTS["general"]
    }


# def select_model(query_analysis: dict) -> str:
#     """Select appropriate model based on query requirements"""
#     if query_analysis["type"] == "medical":
#         return "claude-3-5-sonnet-20241022"  # Highest accuracy
#     elif query_analysis["type"] == "math":
#         return "gpt-4o-mini"  # Good balance
#     else:
#         return "grok-2-1212"  # Most cost-effective

def select_model(query_analysis):
    """Select the most efficient model based on requirements: cost, accuracy, latency, CO₂."""
    best_model = None
    best_score = float("inf")  # Lower is better

    for model, metadata in MODEL_METADATA.items():
        model_score = (
            (1 / query_analysis["min_accuracy"]) * 10  # Accuracy weight
            + metadata["base_cost_per_token"] * 1000  # Cost weight
            + metadata["power_consumption_factor"] * 5  # Sustainability weight
        )
        
        # Find the model with the lowest score
        if model_score < best_score:
            best_score = model_score
            best_model = model

    return best_model



# Add this class after your imports
class QueryMetrics:
    def __init__(self):
        self.history = []
    
    def add_metric(self, model: str, metrics: Dict):
        timestamp = datetime.now().isoformat()
        self.history.append({
            "timestamp": timestamp,
            "model": model,
            **metrics
        })
        
        # Keep only last 100 queries for memory efficiency
        if len(self.history) > 100:
            self.history.pop(0)
    
    def get_model_stats(self, model: str) -> Dict:
        model_queries = [q for q in self.history if q["model"] == model]
        if not model_queries:
            return {}
        
        return {
            "avg_inference_time": sum(q["inference_time_seconds"] for q in model_queries) / len(model_queries),
            "avg_co2": sum(q["carbon_emissions"] for q in model_queries) / len(model_queries),
            "total_queries": len(model_queries),
            "total_tokens": sum(q.get("total_tokens", 0) for q in model_queries),
            "📧 Emails Offset": kg_co2 * 100000  # ✅ Ensure this is always included

        }
    
    def get_all_stats(self) -> Dict:
        return {model: self.get_model_stats(model) for model in MODEL_METADATA.keys()}

# Initialize metrics tracker globally
query_metrics = QueryMetrics()

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


def co2_equivalency(kg_co2: float) -> dict:
    """Convert CO₂ savings into real-world impact."""
    # return {
    #     "🌳 Trees Saved": kg_co2 * 0.05,
    #     "💡 Hours of LED Bulb Power": kg_co2 * 17,
    #     "✈️ Airplane Miles Avoided": kg_co2 * 2,
    #    
    #     "🎮 Gaming PC Usage Reduced (hrs)": kg_co2 * 3,
    # }
    
    # def co2_equivalency(kg_co2: float) -> dict:
    """Convert CO₂ savings into widely accepted sustainability metrics."""
    return {
        "🏡 Home Electricity Saved (hrs)": kg_co2 * 3.6,
        "🚲 Bike Rides Instead of Car Trips": kg_co2 / 0.15,
        "🌊 Plastic Bottles Not Produced": kg_co2 / 0.08,
        "📧 Emails Offset": kg_co2 * 100000  # Fun fact!
    }

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
    print("Open ai response",response.choices[0].message.content)
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
    print("deep seek",response.choices[0].message.content)
    return response.choices[0].message.content
    

def anthropic_ai_call(prompt):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    print("Anthropic",message.content)
    return message.content[0].text



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
    print("XAI: ",completion.choices[0].message.content)
    # print("XAI tokens: ",completion.choices[0].)
    return completion.choices[0].message.content

@app.route("/metrics", methods=["GET"])
def get_metrics():
    stats = query_metrics.get_all_stats()
    
    # Compute CO₂ equivalency
    total_co2_saved = sum(m["avg_co2"] for m in stats.values() if m)
    # impact = co2_equivalency(total_co2_saved)
    
    # Ensure real_world_impact is always present
    impact = co2_equivalency(total_co2_saved) if total_co2_saved > 0 else {
        "🏡 Home Electricity Saved (hrs)": 0,
        "🚲 Bike Rides Instead of Car Trips": 0,
        "🌊 Plastic Bottles Not Produced": 0,
        "📧 Emails Offset": 0  # ✅ Matches the new streamlined metrics
    }

    return jsonify({
        "current_stats": stats,
        "model_metadata": MODEL_METADATA,
        "total_co2_saved": total_co2_saved,
        "real_world_impact": impact
    })


@app.route("/process_prompt",methods=["POST"])
def process_prompt():
    # data = request.json
    # prompt = data.get("prompt", "")
    # model = data.get("model", "gpt-4")

    # estimated_tokens = count_tokens(prompt)

    # if model == "deepseek-chat":
    #     start_time = time.time()
    #     response = deepseek_ai_call(prompt)
    #     end_time = time.time()
    # else:  # default OpenAI
    #     start_time = time.time()
    #     response = open_ai_call(prompt)
    #     end_time = time.time()

    # inference_time = end_time - start_time
    # co2_emissions = calculate_co2_emissions(inference_time)

    # return jsonify({
    #     "prompt": prompt,
    #     "token_count": estimated_tokens,
    #     "model": model,
    #     "carbon_emissions": co2_emissions,
    #     "response": response,
    #     "inference_time_seconds": inference_time
    # })

    data = request.json
    prompt = data.get("prompt", "")
    
     # Analyze query
    query_analysis = analyze_query(prompt)
    
    # Select single best model
    selected_model = select_model(query_analysis)
    

    estimated_tokens = count_tokens(prompt)

    models = {
        "gpt-4o-mini": open_ai_call,
        # "deepseek-chat": deepseek_ai_call,
        "claude-3-5-sonnet-20241022": anthropic_ai_call,
        "grok-2-1212": xai_ai_call,
    }

    # Only call the selected model
    model_function = models[selected_model]
    
    # Perform inference and calculate carbon emissions for each model
    results = {}
    
    try:
        # Make single API call to selected model
        start_time = time.time()
        response = model_function(prompt)
        end_time = time.time()
        
        inference_time = end_time - start_time
        co2_emissions = calculate_co2_emissions(inference_time)
        
        # Calculate cost using model metadata
        cost = estimated_tokens * MODEL_METADATA[selected_model]["base_cost_per_token"]
        
        # Add to metrics history
        query_metrics.add_metric(selected_model, {
            "inference_time_seconds": inference_time,
            "carbon_emissions": co2_emissions,
            "total_tokens": estimated_tokens,
            "cost": cost
        })
        
        real_world_impact = co2_equivalency(co2_emissions) if co2_emissions > 0 else {
        "🌳 Trees Saved": 0,
        "💡 Hours of LED Bulb Power": 0,
        "✈️ Airplane Miles Avoided": 0,
        "📧 Emails Offset": 0,
        "🎮 Gaming PC Usage Reduced (hrs)": 0,
        }
        
        return jsonify({
            "prompt": prompt,
            "token_count": estimated_tokens,
            "query_analysis": {
                "type": query_analysis["type"],
                "required_accuracy": query_analysis["min_accuracy"],
                "cost_sensitivity": query_analysis["cost_sensitivity"],
                "max_latency": query_analysis["max_latency"]
            },
            "model_selection": {
                "selected_model": selected_model,
                "power_consumption_factor": MODEL_METADATA[selected_model]["power_consumption_factor"],
                "base_cost_per_token": MODEL_METADATA[selected_model]["base_cost_per_token"]
            },
            "performance": {
                "response": response,
                "inference_time_seconds": inference_time,
                "carbon_emissions": co2_emissions,
                "cost": cost
            },
            "real_world_impact": real_world_impact 
        })
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "query_analysis": query_analysis,
            "selected_model": selected_model
        }), 500
    

    # for model_name, model_function in models.items():
    #     try:
    #         start_time = time.time()
    #         response = model_function(prompt)
    #         end_time = time.time()
    #         inference_time = end_time - start_time
    #         co2_emissions = calculate_co2_emissions(inference_time)

    #         results[model_name] = {
    #             "response": response,
    #             "inference_time_seconds": inference_time,
    #             "carbon_emissions": co2_emissions,
    #         }
    #         print(results[model_name])
    #     except Exception as e:
    #         results[model_name] = {
    #             "response": f"Error: {str(e)}",
    #             "inference_time_seconds": None,
    #             "carbon_emissions": None,
    #         }

    # return jsonify({
    #     "prompt": prompt,
    #     "token_count": estimated_tokens,
    #     "results": results
    # })


if __name__ == "__main__":
    app.run(debug=True)
    
    
