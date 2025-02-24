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


MODEL_METADATA = {
    "gpt-4o-mini": {
        "base_cost_per_token": 0.00003,
        "max_tokens": 128000,
        "typical_speed": "medium-fast",
        "power_consumption_factor": 1.5
    },
    "claude-3-5-sonnet-20241022": {
        "base_cost_per_token": 0.000003,
        "max_tokens": 200000,
        "typical_speed": "fast",
        "power_consumption_factor": 1.2
    },
    "grok-2-1212": {
        "base_cost_per_token": 0.00001,
        "max_tokens": 32768,
        "typical_speed": "medium",
        "power_consumption_factor": 1.8
    }
}


MODEL_METRICS = {
    "gpt-4o-mini": {
        "accuracy": 0.94,
        "completion_tokens": 300,
        "inference_time": 1.2,
        "co2_emissions": 0.02,
        "cost_per_1000_tokens": 0.03
    },
    "claude-3-5-sonnet-20241022": {
        "accuracy": 0.93,
        "completion_tokens": 280,
        "inference_time": 1.0,
        "co2_emissions": 0.018,
        "cost_per_1000_tokens": 0.002
    },
    "grok-2-1212": {
        "accuracy": 0.85,
        "completion_tokens": 200,
        "inference_time": 0.8,
        "co2_emissions": 0.015,
        "cost_per_1000_tokens": 0.001
    }
}



# Future : Use BERT or other NLP techniques for this
QUERY_REQUIREMENTS = {
    "medical": {
        "min_accuracy": 0.90,
        "max_latency": 3.0,
        "cost_sensitivity": "low",
        "keywords": [ "diagnosis", "medical", "health", "symptoms", "disease", "treatment",
            "prescription", "medicine", "vaccine", "surgery", "mental health", "doctor",
            "cancer", "infection", "diabetes", "cardiology", "neurology"]
    },
    "math": {
        "min_accuracy": 0.85,
        "max_latency": 5.0,
        "cost_sensitivity": "medium",
        "keywords": ["calculate", "solve", "equation", "math", "formula", 
            "geometry", "algebra", "probability", "integral", "derivative",
            "linear regression", "matrix", "calculus", "graph theory"]
    },
    "finance": {  
        "min_accuracy": 0.88,
        "max_latency": 6.0,
        "cost_sensitivity": "medium",
        "keywords": [
            "stock", "market", "investment", "interest rate", "inflation", 
            "portfolio", "bitcoin", "crypto", "loan", "debt", "financial",
            "trading", "forex", "GDP", "economy", "banking", "exchange rate"
        ]
    },
    "technology": { 
        "min_accuracy": 0.80,
        "max_latency": 7.0,
        "cost_sensitivity": "high",
        "keywords": [
            "AI", "machine learning", "deep learning", "NLP", "chatbot",
            "cloud computing", "cybersecurity", "big data", "IoT", "blockchain",
            "quantum computing", "data science", "LLM", "GPU", "neural network"
        ]
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



def select_model(query_analysis):
   
    query_type = query_analysis["type"]
    pareto_models = []

    # Step 1: Find Pareto-optimal models (not dominated)
    for model, metrics in MODEL_METRICS.items():
        dominated = False
        for other_model, other_metrics in MODEL_METRICS.items():
            if other_model != model:
                # Check if this model is strictly worse than another
                if (
                    other_metrics["accuracy"] >= metrics["accuracy"]
                    and other_metrics["cost_per_1000_tokens"] <= metrics["cost_per_1000_tokens"]
                    and other_metrics["co2_emissions"] <= metrics["co2_emissions"]
                ):
                    dominated = True
                    break  # No need to check further

        if not dominated:
            pareto_models.append((model, metrics))

    # Choose the best tradeoff from Pareto-optimal models
    if query_type in ["medical", "math"]:
        best_model = min(pareto_models, key=lambda x: (1 - x[1]["accuracy"]))[0]  # Highest accuracy
    if query_type == "technology":
        best_model = "claude-3-5-sonnet-20241022" 
    else:
        best_model = min(pareto_models, key=lambda x: (x[1]["cost_per_1000_tokens"] + x[1]["co2_emissions"]))[0]  # Greenest + Cheapest

    print(f"Selected Pareto-Optimized Model for {query_type}: {best_model}")
    return best_model

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

    # def co2_equivalency(kg_co2: float) -> dict:
    """Convert CO₂ savings into widely accepted sustainability metrics."""
    return {
        "🏡 Home Electricity Saved (hrs)": kg_co2 / 0.25,
        "🚲 Bike Rides Instead of Car Trips": kg_co2 / 0.15,
        "🌊 Plastic Bottles Not Produced": kg_co2 / 0.08,
        "📧 Emails Offset": kg_co2 * 50000  # Fun fact!
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
                "base_cost_per_token": MODEL_METADATA[selected_model]["base_cost_per_token"],
                "completion_tokens": MODEL_METRICS[selected_model].get("completion_tokens", 0),
                "inference_time": MODEL_METRICS[selected_model].get("inference_time", 0),
                "co2_emissions": MODEL_METRICS[selected_model].get("co2_emissions", 0),
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
    


if __name__ == "__main__":
    app.run(debug=True)
    
    