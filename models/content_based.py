import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

def load_categories(file_path="data/categories.json"):
    with open(file_path, "r") as f:
        return json.load(f)

def recommend_content_based(history, top_n=10, temperature=.7):
    """
    Args:
        history: List of category names
        top_n: Number of recommendations to return
        temperature: Float between 0.0-1.0 controlling randomness
                    0.0 = deterministic
                    1.0 = most random
    """
    categories = load_categories()
    descriptions = list(categories.values())
    category_names = list(categories.keys())
    
    print(f"Number of categories loaded: {len(category_names)}")
    
    valid_history = [cat for cat in history if cat in category_names]
    
    if len(valid_history) != len(history):
        missing_categories = set(history) - set(valid_history)
        print(f"Warning: The following categories were not found: {missing_categories}")
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    indices = [category_names.index(cat) for cat in valid_history]
    avg_similarity = similarity_matrix[indices].mean(axis=0)
    
    # Scale random noise by temperature
    noise_magnitude = temperature * 0.5  # Max noise = half of similarity
    random_noise = np.random.uniform(-noise_magnitude, noise_magnitude, len(avg_similarity))
    avg_similarity = avg_similarity + random_noise
    
    # Get larger pool for more variety
    pool_size = min(int(top_n * (1 + temperature * 3)), len(category_names))
    recommended_indices = avg_similarity.argsort()[-pool_size:][::-1]
    
    recommendations_pool = [category_names[i] for i in recommended_indices if category_names[i] not in valid_history]
    final_recommendations = random.sample(recommendations_pool[:pool_size], min(top_n, len(recommendations_pool)))
    
    return final_recommendations
