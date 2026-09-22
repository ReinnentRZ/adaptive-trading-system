import pandas as pd

def klines_to_df(all_candles_list):

    df = pd.DataFrame(all_candles_list)

    for col in ["id", "symbol", "interval", "ignore"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    float_columns = [
        "open", "high", "low", "close", 
        "volume", "quote_volume", 
        "taker_buy_base", "taker_buy_quote"
    ]
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    int_columns = [
        "time_open", "time_closed", 
        "first_trade_id", "last_trade_id", 
        "trades_count"
    ]
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    if "is_closed" in df.columns:
        df["is_closed"] = df["is_closed"].astype(bool)
    
    return df

def klines_to_chart(klines):
    columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base", "Taker Buy Quote", "Ignore"
    ]
    
    df = pd.DataFrame(klines, columns=columns)
    
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    
    return df