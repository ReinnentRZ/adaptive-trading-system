import json
import uuid

def process_kline(message):
    parsed = json.loads(message)
    k = parsed["k"]

    data = {
        "id": uuid.uuid4().hex,
        "symbol": k["s"],
        "interval": k["i"],
        "close": float(k["c"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "time": int(k["t"])
    }

    print("log:", data)
    return data