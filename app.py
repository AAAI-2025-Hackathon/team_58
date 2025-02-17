from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to EcoGPT!"


@app.route("/process_prompt",methods=["POST"])
def process_prompt():
    data = request.json
    prompt = data.get("prompt","")
    
    token_count=len(prompt.split())
    
    return jsonify({
        "prompt":prompt,
        "token_count":token_count
    })

if __name__ == "__main__":
    app.run(debug=True)