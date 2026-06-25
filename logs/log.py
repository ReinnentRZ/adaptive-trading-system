from datetime import datetime
from src.feeds.stream import process_kline
def log_close(data_kline, time_close, rsi_value, rsi_smoothing, adx_value, adx_smoothing):
    harga_close = data_kline["close"]

    waktu = datetime.fromtimestamp(time_close / 1000).strftime('%Y-%m-%d %H:%M:%S')
    log_format =f"[{waktu}]  Close: {harga_close:<10.2f}  |  RSI: {rsi_value:<6.2f}  |  RSI Smoothing: {rsi_smoothing:<6.2f}  |  ADX: {adx_value:<6.2f}  |  ADX Smoothing: {adx_smoothing:<6.2f}"
    print(log_format)

    with open("logs/logs.txt", "a") as f:
        f.write(log_format + "\n")


