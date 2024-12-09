import requests

response = requests.get("http://localhost:5000/api/categories", 
                       params={
                           "top_n": 5,
                           "temperature": 0.7
                       })
data = response.json()
recommendations = data["recommendations"]
print(recommendations)