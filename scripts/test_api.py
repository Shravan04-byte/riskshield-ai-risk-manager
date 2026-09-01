import requests

BASE = "http://localhost:8000"

examples = requests.get(f"{BASE}/example-transactions").json()["examples"]
fp = next(e for e in examples if e["category"] == "false_positive")

print("Posting false_positive transaction...")
response = requests.post(f"{BASE}/assess-transaction", json=fp["transaction"])
print(f"Status: {response.status_code}")
print(response.json())