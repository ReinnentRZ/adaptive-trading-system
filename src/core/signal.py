from dataclasses import dataclass
from src.core.enums import OrderType

@dataclass
class Signal:
    type: OrderType      
    candle_close: float 
    confidence: float    # Tingkat keyakinan dalam persen (e.g., 87.5%)
    raw_vote: int        # Angka voting asli dari model (e.g., +6 atau -5)

    @classmethod
    def from_lorentzian(cls, prediction_result, candle_close, neighbors_count=8):
        lorentz_name = prediction_result.signal.name 
        
        if lorentz_name == "HOLD":
            return None
            
        if lorentz_name == "LONG":
            order_type = OrderType.LONG
        elif lorentz_name == "SHORT":
            order_type = OrderType.SHORT
        else:
            return None 

        raw_vote = int(prediction_result.prediction)
        

        confidence = (abs(raw_vote) / neighbors_count) * 100.0
        confidence = round(confidence, 2) 

        print(f"{order_type.name} Berhasil dibuat | Close: {candle_close} | Vote: {raw_vote} | Conf: {confidence}%")

        return cls(
            type=order_type,
            candle_close=float(candle_close), 
            confidence=confidence,
            raw_vote=raw_vote
        )