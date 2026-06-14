import websocket
from src.feeds.stream import process_kline

def on_message(ws, message):
    data = process_kline(message)
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws, close_msg):
    print("closed")

def streamKline(symbol, interval):
    socket = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"

    ws = websocket.WebSocketApp(
        socket,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()