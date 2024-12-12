import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class CategoryRecommender:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Initialize preprocessor
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Advanced text preprocessing with lemmatization and stop word removal
        """
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        
        # Lemmatize, remove stop words and non-alphabetic tokens
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalpha() and token not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)

    def load_data(self, categories_path='data/categories.json', feedback_path='data/feedback.json'):
        """
        Load and preprocess categories and feedback data
        """
        # Load categories
        with open(categories_path, 'r') as f:
            categories = json.load(f)
        
        # Load feedback
        with open(feedback_path, 'r') as f:
            feedback = json.load(f)
        
        # Preprocess category descriptions
        category_names = list(categories.keys())
        descriptions = [self.preprocess_text(desc) for desc in categories.values()]
        
        # Create labels based on feedback
        feedback_dict = {item["category"]: item["liked"] for item in feedback}
        labels = [feedback_dict.get(cat, False) for cat in category_names]
        
        return descriptions, labels, category_names

    def create_model_pipeline(self):
        """
        Create a machine learning pipeline with TF-IDF vectorization and RandomForest
        """
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),  # Use unigrams and bigrams
                max_features=5000,    # Limit features
                stop_words='english'  # Additional stop word filtering
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',  # Handle potential class imbalance
                random_state=42
            ))
        ])

    def train_and_evaluate(self, descriptions, labels):
        """
        Train the model and provide detailed evaluation
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            descriptions, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Create and train the pipeline
        pipeline = self.create_model_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Detailed evaluation
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, descriptions, labels, cv=5)
        print(f"\nCross-validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return pipeline

    def recommend_categories(self, pipeline, descriptions, category_names, top_n=10, temperature=0.7):
        """
        Recommend categories with probabilistic sampling
        """
        # Get prediction probabilities
        probas = pipeline.predict_proba(descriptions)[:, 1]
        
        # Temperature-based sampling
        if temperature > 0:
            adjusted_probs = probas ** (1 / temperature)
            adjusted_probs /= adjusted_probs.sum()
            recommended_indices = np.random.choice(
                len(category_names), 
                size=min(top_n, len(category_names)), 
                replace=False, 
                p=adjusted_probs
            )
        else:
            # Deterministic top-n selection
            recommended_indices = np.argsort(probas)[::-1][:top_n]
        
        recommendations = [category_names[i] for i in recommended_indices]
        return recommendations

    def run(self):
        """
        Main execution method
        """
        # Load data
        descriptions, labels, category_names = self.load_data()
        
        # Train and evaluate model
        trained_pipeline = self.train_and_evaluate(descriptions, labels)
        
        # Generate recommendations
        recommendations = self.recommend_categories(
            trained_pipeline, descriptions, category_names
        )
        
        print("\nRecommended Categories:")
        for rec in recommendations:
            print(rec)

# Usage
if __name__ == "__main__":
    recommender = CategoryRecommender()
    recommender.run()