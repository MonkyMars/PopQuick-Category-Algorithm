import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from utils.file_io import load_categories, load_feedback
import nltk

def train_model():
    categories = load_categories()
    category_names = list(categories.keys())
    descriptions = list(categories.values())

    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', stop_words=stop_words)
    X = vectorizer.fit_transform(descriptions)

    # Create labels for all categories
    feedback = load_feedback()
    feedback_dict = {item["category"]: item["liked"] for item in feedback}
    y = np.array([feedback_dict.get(cat, False) for cat in category_names])

    # Model selection and hyperparameter tuning
    model = LogisticRegression()
    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)

    # Model evaluation
    y_pred = grid_search.predict(X)
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 score:", f1_score(y, y_pred))

    with open("models/model.pkl", "wb") as f:
        pickle.dump((vectorizer, grid_search.best_estimator_), f)

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