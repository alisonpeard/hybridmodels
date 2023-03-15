"""Authenticate via basic HTTP with Python."""
import os
import json
import requests

wd = os.getcwd()

os.environ['PL_API_KEY'] = 'PLAK5d7f0427feda4778b1f8ad7aec70ce0a'
PLANET_API_KEY = os.getenv('PL_API_KEY')
BASE_URL = "https://api.planet.com/data/v1"

session = requests.Session()
session.auth = (PLANET_API_KEY, "")
res = session.get(BASE_URL)
# print(res.status_code)
# print(res.text)

f = open(os.path.join(wd, "emnati-madagascar.json"))
json_request = json.load(f)

req = session.post("https://api.planet.com/compute/ops/orders/v2", json=json_request)
print(req.status_code)
print(req.text)
import pdb; pdb.set_trace()