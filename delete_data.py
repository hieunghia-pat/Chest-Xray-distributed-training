import redis
import json
from tqdm import tqdm

xstk_data = json.load(open("XSTK.json"))
dstt_data = json.load(open("DSTT.json"))

XSTK_db = "XSTK"
DSTT_db = "DSTT"

server = redis.Redis(host="localhost", port="6379")

for id in tqdm(xstk_data):
    student = xstk_data[id]
    server.delete(f"{XSTK_db}:{id}")

server.srem(f"{XSTK_db}")

for id in tqdm(dstt_data):
    student = dstt_data[id]
    server.delete(f"{DSTT_db}:{id}")
server.srem(f"{DSTT_db}")