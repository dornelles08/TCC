import requests

r = requests.get("https://parallelum.com.br/fipe/api/v2/references",
                 headers={"Accept": "application/json"})

print(r.text)
