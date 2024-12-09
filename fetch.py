import requests

response = requests.get("http://localhost:5000/api/categories", params={"top_n": 10, "temperature": 0}) # higher temperature = more randomness, and vise versa.
data = response.json()
recommendations = data["recommendations"]
print(recommendations)