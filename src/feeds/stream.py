import json
import uuid

def process_kline(message):
    parsed = json.loads(message)
    k = parsed["k"]

    return {
        "id": uuid.uuid4().hex,
        "symbol": k["s"],
        "interval": k["i"],
        "close": float(k["c"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "time_open": int(k["t"]),
        "time_closed" : int(k["T"]),
        "is_closed": k["x"]
        

    }
