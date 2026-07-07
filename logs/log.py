import os
import psutil
from datetime import datetime

def log_close(
    data_kline, time_close, 
    rsi_value, rsi_smoothing, 
    adx_value, adx_smoothing, 
    cci_value, cci_smoothing, 
    wt1_value, wt2_value, 
    array_tetangga, raw_prediction, signal_name
):
    # 1. Konversi Data Dasar
    harga_close = float(data_kline["close"])
    waktu = datetime.fromtimestamp(time_close / 1000).strftime('%Y-%m-%d %H:%M:%S')
    
    # 2. Pengecekan RAM Usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)

    # 3. Handling Nilai None untuk Indikator
    rsi_val = float(rsi_value) if rsi_value is not None else 0.0
    rsi_smth = float(rsi_smoothing) if rsi_smoothing is not None else 0.0
    adx_val = float(adx_value) if adx_value is not None else 0.0
    adx_smth = float(adx_smoothing) if adx_smoothing is not None else 0.0
    cci_val = float(cci_value) if cci_value is not None else 0.0
    cci_smth = float(cci_smoothing) if cci_smoothing is not None else 0.0
    wt1_val = float(wt1_value) if wt1_value is not None else 0.0
    wt2_val = float(wt2_value) if wt2_value is not None else 0.0
    
    # 4. Format Teks Sinyal dan Vote
    if signal_name == "LONG":
        signal_text = "BUY "
    elif signal_name == "SHORT":
        signal_text = "SELL"
    else:
        signal_text = "HOLD"

    vote_text = f"+{raw_prediction}" if raw_prediction > 0 else f"{raw_prediction}"

    # -------------------------------------------------------------
    # HASIL 1: FORMAT UNTUK TERMINAL (Ringkas & Enak Dilihat)
    # -------------------------------------------------------------
    terminal_format = (
        f"[{waktu}]  "
        f"Close: {harga_close:<9.2f}  |  "
        #f"RSI: {rsi_val:<6.2f}  |  "
        #f"ADX: {adx_val:<6.2f}  |  "
        #f"CCI: {cci_val:<8.2f}  |  "
        #f"WT1: {wt1_val:<7.2f}  |  "
        f"VOTE: {vote_text:<4}  |  "
        f"SIGNAL: {signal_text}  |  "
        f"RAM: {ram_usage:.1f} MB"
    )
    print(terminal_format)

    # -------------------------------------------------------------
    # HASIL 2: FORMAT UNTUK LOGS.TXT (Lengkap dengan data Smoothing)
    # -------------------------------------------------------------
    file_log_format = (
        f"[{waktu}]  "
        f"Close: {harga_close:<9.2f}  |  "
        f"RSI: {rsi_val:.2f}(S:{rsi_smth:.2f})  |  "
        f"ADX: {adx_val:.2f}(S:{adx_smth:.2f})  |  "
        f"CCI: {cci_val:.2f}(S:{cci_smth:.2f})  |  "
        f"WT1/WT2: {wt1_val:.2f}/{wt2_val:.2f}  |  "
        f"NEIGHBOURS: {str(array_tetangga)}  |  "
        f"VOTE: {vote_text}  |  "
        f"SIGNAL: {signal_text}  |  "
        f"RAM: {ram_usage:.1f} MB"
    )

    # Memastikan folder logs ada sebelum menulis file agar tidak error
    os.makedirs("logs", exist_ok=True)
    
    with open("logs/logs.txt", "a", encoding="utf-8") as f:
        f.write(file_log_format + "\n")