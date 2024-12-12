import requests

response = requests.get(
    "http://localhost:5000/api/categories", params={"top_n": 10, "temperature": 0.1}
)
data = response.json()
if response.status_code == 200:
    try:
        recommendations = data["recommendations"]
        print(recommendations)
    except KeyError:
        print(data["message"], data["status"])
else:
    print(data)
