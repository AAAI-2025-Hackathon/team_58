from flask import Flask, request, jsonify
import os
import openai
import tiktoken

app = Flask(__name__)

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY


def count_tokens(prompt):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(prompt))
        return num_tokens
    except:
        print("Error: Could not encode prompt")
        return len(prompt.split())



@app.route("/")
def home():
    return "Welcome to EcoGPT!"


@app.route("/process_prompt",methods=["POST"])
def process_prompt():
    data = request.json
    prompt = data.get("prompt","")
    model = data.get("model","gpt-4")
    estimated_tokens = count_tokens(prompt)         
    return jsonify({
            "prompt":prompt,
            "token_count":estimated_tokens,
    })

if __name__ == "__main__":
    app.run(debug=True)