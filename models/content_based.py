import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

def load_categories(file_path="data/categories.json"):
    with open(file_path, "r") as f:
        return json.load(f)

def load_feedback(file_path="data/feedback.json"):
    with open(file_path, "r") as f:
        return json.load(f)

def train_model():
    categories = load_categories()
    category_names = list(categories.keys())
    descriptions = list(categories.values())

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(descriptions)

    feedback = load_feedback()
    feedback_categories = [item['category'] for item in feedback if item['category'] in category_names]
    y = np.array([1 if item['liked'] else 0 for item in feedback if item['category'] in category_names])

    feedback_indices = [category_names.index(cat) for cat in feedback_categories]
    X_feedback = X[feedback_indices]

    model = LogisticRegression()
    model.fit(X_feedback, y)

    with open("models/model.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)

def recommend_ml(top_n=10, temperature=0.7):
    with open("models/model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)

    categories = load_categories()
    category_names = list(categories.keys())
    descriptions = list(categories.values())

    X = vectorizer.transform(descriptions)
    predictions = model.predict_proba(X)[:, 1]

    if temperature > 0:
        # Adjust probabilities by temperature
        adjusted_probs = predictions ** (1 / temperature)
        adjusted_probs /= adjusted_probs.sum()
        recommended_indices = np.random.choice(len(category_names), size=top_n, replace=False, p=adjusted_probs)
    else:
        # For temperature=0, return the top_n highest scores
        recommended_indices = np.argsort(predictions)[::-1][:top_n]

    recommendations = [category_names[i] for i in recommended_indices]

    return recommendations


def recommend_content_based(liked_categories, disliked_categories=None, top_n=10, temperature=0.7):
    """
    Enhanced algorithm to provide recommendations closely tied to user's preferences.

    Args:
        liked_categories: List of category names the user likes.
        disliked_categories: List of category names the user dislikes.
        top_n: Number of recommendations to return.
        temperature: Float between 0.0-1.0 controlling randomness.
    """
    categories = load_categories()
    descriptions = list(categories.values())
    category_names = list(categories.keys())

    # Vectorize the category descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Ensure disliked_categories is a list
    print(disliked_categories, liked_categories, top_n, temperature)
    if disliked_categories is None:
        disliked_categories = []

    # Indices of liked and disliked categories
    liked_indices = [category_names.index(cat) for cat in liked_categories if cat in category_names]
    disliked_indices = [category_names.index(cat) for cat in disliked_categories if cat in category_names]

    if not liked_indices:
        raise ValueError("No valid liked categories found.")

    # Compute the positive and negative user profiles
    positive_profile = tfidf_matrix[liked_indices].mean(axis=0)

    # If there are dislikes, subtract them from the positive profile
    if disliked_indices:
        negative_profile = tfidf_matrix[disliked_indices].mean(axis=0)
        user_profile = positive_profile - negative_profile
    else:
        user_profile = positive_profile

    # Ensure the user_profile is a 1D array
    user_profile = user_profile.A.flatten()

    # Compute similarities between user profile and all categories
    similarity_scores = tfidf_matrix.dot(user_profile)

    # Exclude liked and disliked categories from recommendations
    excluded_indices = set(liked_indices + disliked_indices)
    similarity_scores[list(excluded_indices)] = -np.inf

    # Apply temperature scaling
    if temperature > 0:
        probabilities = np.exp(similarity_scores / temperature)
        probabilities /= probabilities.sum()
        recommended_indices = np.random.choice(len(category_names), size=top_n, replace=False, p=probabilities)
    else:
        # For temperature=0, return the top_n highest scores
        recommended_indices = np.argsort(similarity_scores)[::-1][:top_n]

    recommendations = [category_names[i] for i in recommended_indices]

    return recommendations
