import json
import uuid

def process_kline(message):

    parsed = json.loads(message)
    k = parsed["k"]

    return {
        "id": uuid.uuid4().hex,
        "symbol": k["s"],
        "interval": k["i"],
        "time_open": int(k["t"]),
        "time_closed": int(k["T"]),
        "first_trade_id": int(k["f"]),
        "last_trade_id": int(k["L"]),
        "open": float(k["o"]),
        "close": float(k["c"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "volume": float(k["v"]),
        "trades_count": int(k["n"]),
        "is_closed": bool(k["x"]),
        "quote_volume": float(k["q"]),
        "taker_buy_base": float(k["V"]),
        "taker_buy_quote": float(k["Q"]),
        "ignore": str(k["B"])
    }