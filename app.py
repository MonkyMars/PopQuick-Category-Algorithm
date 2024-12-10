from models.content_based import recommend_content_based, train_model, recommend_ml
import json
from flask import Flask, request, jsonify
from utils.file_io import load_json
app = Flask(__name__)

# Load the feedback data from the JSON file
def load_feedback(file_path="data/feedback.json"):
    with open(file_path, "r") as f:
        return json.load(f)

# Convert the feedback data to user_history format
def convert_to_user_history(feedback):
    # Only include categories that were liked
    return [item["category"] for item in feedback if item["liked"]]

@app.route('/api/categories', methods=['GET'])
def get_recommendations():
    try:
        top_n = request.args.get('top_n', default=10, type=int)
        temperature = request.args.get('temperature', default=.7, type=float)
        model = request.args.get('model', default='ml', type=str)
        
        if model == 'ml':
            train_model()
            recommendations = recommend_ml(top_n=top_n, temperature=temperature)
        elif model == 'content_based':
            feedback = load_feedback()
            user_history = convert_to_user_history(feedback)
            recommendations = recommend_content_based(user_history, top_n=top_n, temperature=temperature)
            
        return jsonify({
            "status": "success",
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
