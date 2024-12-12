import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from utils.file_io import load_categories, load_feedback


def train_model():
    categories = load_categories()
    category_names = list(categories.keys())
    descriptions = list(categories.values())

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(descriptions)

    # Create labels for all categories
    feedback = load_feedback()
    feedback_dict = {item["category"]: item["liked"] for item in feedback}
    y = np.array([feedback_dict.get(cat, False) for cat in category_names])

    model = LogisticRegression()
    model.fit(X, y)

    with open("models/model.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)


def recommend_ml(top_n=10, temperature=0.7, vectorizer=None, model=None):
    categories = load_categories()
    category_names = list(categories.keys())
    descriptions = list(categories.values())

    X = vectorizer.transform(descriptions)
    predictions = model.predict_proba(X)[:, 1]

    if temperature > 0:
        adjusted_probs = predictions ** (1 / temperature)
        adjusted_probs /= adjusted_probs.sum()
        recommended_indices = np.random.choice(
            len(category_names), size=top_n, replace=False, p=adjusted_probs
        )
    else:
        recommended_indices = np.argsort(predictions)[::-1][:top_n]

    recommendations = [category_names[i] for i in recommended_indices]

    return recommendations

