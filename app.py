from flask import Flask, request, jsonify
import pickle
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.popAI import train_model, recommend_ml
from models.QuickAI import CategoryRecommender
from utils.file_io import load_categories, load_feedback

app = Flask(__name__)

class ModelManager:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.category_recommender = None
        
    def initialize_model(self):
        """
        Initialize the recommendation model
        """
        try:
            # Train the model first
            train_model()
            
            # Load the trained model
            with open("models/model.pkl", "rb") as f:
                self.vectorizer, self.model = pickle.load(f)
            
            # Initialize the category recommender
            self.category_recommender = CategoryRecommender()
            
            return True
        except Exception as e:
            print(f"Model initialization error: {e}")
            return False

# Global model manager
model_manager = ModelManager()
model_initialized = model_manager.initialize_model()

@app.route("/api/categories", methods=["GET"])
def get_recommendations():
    if not model_initialized:
        return jsonify({
            "status": "error", 
            "message": "Model not properly initialized"
        }), 500
    
    try:
        # Get parameters with default values
        top_n = request.args.get("top_n", default=10, type=int)
        temperature = request.args.get("temperature", default=0.7, type=float)
        
        # Option 1: Using recommend_ml from popAI
        popai_recommendations = recommend_ml(
            top_n=top_n, 
            temperature=temperature, 
            vectorizer=model_manager.vectorizer, 
            model=model_manager.model
        )
        
        # Option 2: Using CategoryRecommender from QuickAI
        categories, labels, category_names = model_manager.category_recommender.load_data()
        quickai_recommendations = model_manager.category_recommender.recommend_categories(
            model_manager.category_recommender.create_model_pipeline().fit(categories, labels), 
            categories, 
            category_names, 
            top_n=top_n, 
            temperature=temperature
        )
        
        return jsonify({
            "status": "success", 
            "popai_recommendations": popai_recommendations,
            "quickai_recommendations": quickai_recommendations
        })

    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_initialized": model_initialized
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)