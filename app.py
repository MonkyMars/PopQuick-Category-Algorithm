from models.popAI import train_model, recommend_ml
import json
from flask import Flask, request, jsonify
from utils.file_io import load_json
import pickle
from utils.file_io import load_feedback, convert_to_user_history

app = Flask(__name__)

vectorizer = None
model = None

def initialize_model():
    global vectorizer, model
    train_model()
    with open("models/model.pkl", "rb") as f:
      vectorizer, model = pickle.load(f)

initialize_model()

@app.route("/api/categories", methods=["GET"])
def get_recommendations():
    try:
        top_n = request.args.get("top_n", default=10, type=int)
        temperature = request.args.get("temperature", default=0.7, type=float)
        recommendations = recommend_ml(
            top_n=top_n, temperature=temperature, vectorizer=vectorizer, model=model
        )

        return jsonify({"status": "success", "recommendations": recommendations})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
