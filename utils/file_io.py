import json


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_categories(file_path="data/categories.json"):
    with open(file_path, "r") as f:
        return json.load(f)


def load_feedback(file_path="data/feedback.json"):
    with open(file_path, "r") as f:
        return json.load(f)


def convert_to_user_history(feedback):
    return [item["category"] for item in feedback if item["liked"]]
