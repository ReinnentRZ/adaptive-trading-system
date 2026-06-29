import psutil
import os
from datetime import datetime

def log_close(data_kline, time_close, rsi_value, rsi_smoothing, adx_value, adx_smoothing, cci_value, cci_smoothing, wt1_value, wt2_value, array_tetangga, raw_prediction, signal_name):
    harga_close = data_kline["close"]
    waktu = datetime.fromtimestamp(time_close / 1000).strftime('%Y-%m-%d %H:%M:%S')
    
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)

    if signal_name == "LONG":
        signal_text = "BUY "
    elif signal_name == "SHORT":
        signal_text = "SELL"
    else:
        signal_text = "HOLD"

    vote_text = f"+{raw_prediction}" if raw_prediction > 0 else f"{raw_prediction}"

    log_format = (
        f"[{waktu}]  Close: {harga_close:<10.2f}  |  "
        f"RSI: {rsi_value:<6.2f}  |  ADX: {adx_value:<6.2f}  |  "
        f"CCI: {cci_value:<8.2f}  |  WT1: {wt1_value:<8.2f}  |  "
        f"NEIGHBORS: {str(array_tetangga):<8.2f}  |  "
        f"VOTE: {vote_text:<4}  |  "
        f"SIGNAL: {signal_text}  |  RAM: {ram_usage:.1f} MB"
    )
    print(log_format)

    with open("logs/logs.txt", "a", encoding="utf-8") as f:
        f.write(log_format)