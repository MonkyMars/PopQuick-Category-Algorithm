from models.content_based import recommend_content_based
import json

# Load the feedback data from the JSON file
def load_feedback(file_path="data/feedback.json"):
    with open(file_path, "r") as f:
        return json.load(f)

# Convert the feedback data to user_history format
def convert_to_user_history(feedback):
    # Only include categories that were liked
    return [item["category"] for item in feedback if item["liked"]]

# Main script execution
if __name__ == "__main__":
    feedback = load_feedback()  # Load the feedback data
    user_history = convert_to_user_history(feedback)  # Convert it to user history format
    
    print("\nContent-Based Recommendations:")
    print(recommend_content_based(user_history, top_n=10))
