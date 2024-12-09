from models.content_based import recommend_content_based
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
        # Optional query parameters
        top_n = request.args.get('top_n', default=10, type=int)
        temperature = request.args.get('temperature', default=0.1, type=float)
        
        # Load and process feedback
        feedback = load_feedback()
        user_history = convert_to_user_history(feedback)
        
        # Get recommendations
        feedback = load_json("data/feedback.json")
        liked_categories = [item["category"] for item in feedback if item["liked"]]
        disliked_categories = [item["category"] for item in feedback if not item["liked"]]
        recommendations = recommend_content_based(
            liked_categories=liked_categories,
            disliked_categories=disliked_categories,
            top_n=top_n,
            temperature=temperature
        )
        
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
