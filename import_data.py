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
    for key in student:
        server.hset(
            name = f"{XSTK_db}:{id}",
            key = key,
            value = student[key]
        )
    server.sadd(f"{XSTK_db}", f"{XSTK_db}:{id}")

for id in tqdm(dstt_data):
    student = dstt_data[id]
    for key in student:
        server.hset(
            name = f"{DSTT_db}:{id}",
            key = key,
            value = student[key]
        )
    server.sadd(f"{DSTT_db}", f"{DSTT_db}:{id}")