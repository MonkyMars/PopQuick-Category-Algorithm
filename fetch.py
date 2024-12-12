import requests

response = requests.get(
    "http://localhost:5000/api/categories", params={"top_n": 20, "temperature": .4, "model": 'popai'} # default model is quickai
)
data = response.json()
if response.status_code == 200:
    try:
        recommendations = data['data']
        print(recommendations)
    except KeyError:
        print(data["message"], data["status"])
else:
    print(data)
