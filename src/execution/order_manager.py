import hashlib
import logging
from decimal import Decimal
from typing import Optional

from src.core.enums import OrderType, OrderStatus
from src.core.signal import Signal
from src.config.trading import TRADING
from src.config.strategy import STRATEGY
from src.models.order import Order
from src.models.trade import Trade
from src.execution.broker import Broker
from src.strategies.risk_engine import AdaptiveRiskParams

logger = logging.getLogger("trading_system")


class OrderManager:
    """
    Orchestrator responsible for managing signals, validating risk rules,
    creating orders, and monitoring active positions for Stop Loss and Take Profit breaches.
    Uses Dependency Injection to remain decoupled from specific Broker implementations.
    """
    def __init__(self, broker: Broker):
        self.broker: Broker = broker

    def _get_tick_size(self, symbol: str) -> Decimal:
        """
        Returns the exchange tick size for the symbol.
        Default to Decimal("0.01") for BTCUSDT and others.
        """
        tick_sizes = {
            "BTCUSDT": Decimal("0.01"),
            "ETHUSDT": Decimal("0.01"),
            "BNBUSDT": Decimal("0.1"),
            "SOLUSDT": Decimal("0.01"),
            "ADAUSDT": Decimal("0.0001"),
            "XRPUSDT": Decimal("0.0001"),
            "DOGEUSDT": Decimal("0.00001"),
        }
        return tick_sizes.get(symbol.upper(), Decimal("0.01"))

    def process_signal(
        self,
        signal: Signal,
        symbol: str,
        timestamp: float,
        current_atr: Optional[Decimal] = None,
        adaptive_risk_params: Optional[AdaptiveRiskParams] = None
    ) -> Optional[Order]:
        """
        Processes an incoming trading Signal.
        Decides whether to execute position entries, closures, or reversals.
        """
        # Validate confidence threshold
        if float(signal.confidence) < STRATEGY.confidence_threshold:
            logger.info(
                f"[OrderManager] Signal ignored: Confidence ({signal.confidence}%) "
                f"is below threshold ({STRATEGY.confidence_threshold}%)."
            )
            return None

        active_position = self.broker.get_active_position(symbol)

        if active_position is not None:
            # If we have an active position in the opposite direction of the signal, we close it (position reversal)
            if active_position.direction != signal.type:
                logger.warning(
                    f"[OrderManager] Signal direction ({signal.type.name}) opposite to "
                    f"active position ({active_position.direction.name}). Closing position."
                )
                self.broker.close_active_position(
                    symbol=symbol,
                    exit_price=float(signal.candle_close),
                    timestamp=timestamp
                )
                
                # After closing the old position, open a new one in the direction of the signal
                return self._create_and_execute_order(signal, symbol, timestamp, current_atr, adaptive_risk_params)
            else:
                # Same direction signal - ignore to avoid over-exposure
                logger.info(
                    f"[OrderManager] Hold existing {active_position.direction.name} position. "
                    f"New signal in same direction ignored."
                )
                return None
        else:
            # No active position exists, create a new entry order
            return self._create_and_execute_order(signal, symbol, timestamp, current_atr, adaptive_risk_params)

    def update_market_price(self, symbol: str, current_price: float, timestamp: float) -> None:
        """
        Monitors active positions against Stop Loss and Take Profit thresholds.
        Closes position if risk bounds are breached.
        """
        position = self.broker.get_active_position(symbol)
        if position is None:
            return

        c_price = Decimal(str(current_price))

        # Check ATR-based risk management if enabled and values are set
        if TRADING.use_atr_risk_management and position.sl_price is not None and position.tp_price is not None:
            if position.direction == OrderType.LONG:
                if c_price <= position.sl_price:
                    logger.warning(
                        f"[OrderManager] ATR Stop Loss triggered for {symbol} | "
                        f"Price: {current_price:.2f} | SL: {float(position.sl_price):.2f}"
                    )
                    self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)
                elif c_price >= position.tp_price:
                    logger.warning(
                        f"[OrderManager] ATR Take Profit triggered for {symbol} | "
                        f"Price: {current_price:.2f} | TP: {float(position.tp_price):.2f}"
                    )
                    self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)
            elif position.direction == OrderType.SHORT:
                if c_price >= position.sl_price:
                    logger.warning(
                        f"[OrderManager] ATR Stop Loss triggered for {symbol} | "
                        f"Price: {current_price:.2f} | SL: {float(position.sl_price):.2f}"
                    )
                    self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)
                elif c_price <= position.tp_price:
                    logger.warning(
                        f"[OrderManager] ATR Take Profit triggered for {symbol} | "
                        f"Price: {current_price:.2f} | TP: {float(position.tp_price):.2f}"
                    )
                    self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)
        else:
            # Fallback to percentage-based check
            # Calculate PnL percentage using Decimal precision
            if position.direction == OrderType.LONG:
                pnl_pct = float(((c_price - position.entry_price) / position.entry_price) * Decimal("100.0"))
            elif position.direction == OrderType.SHORT:
                pnl_pct = float(((position.entry_price - c_price) / position.entry_price) * Decimal("100.0"))
            else:
                return

            # Check Stop Loss (SL)
            if pnl_pct <= -TRADING.stop_loss_pct:
                logger.warning(
                    f"[OrderManager] Stop Loss triggered for {symbol} | "
                    f"Price: {current_price:.2f} | PnL: {pnl_pct:.2f}%"
                )
                self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)
            
            # Check Take Profit (TP)
            elif pnl_pct >= TRADING.take_profit_pct:
                logger.warning(
                    f"[OrderManager] Take Profit triggered for {symbol} | "
                    f"Price: {current_price:.2f} | PnL: {pnl_pct:.2f}%"
                )
                self.broker.close_active_position(symbol, exit_price=current_price, timestamp=timestamp)

    def on_candle_close(self, symbol: str, close_price: float, timestamp: float) -> Optional[Trade]:
        """
        Processes candle close event for active position monitoring.
        Increments the holding candle count and triggers time barrier market exit if expired.
        """
        pos = self.broker.increment_position_bars(symbol)
        if pos is not None and pos.check_time_barrier():
            logger.warning(
                f"[OrderManager] Time Barrier expired ({pos.bars_held} >= {pos.max_holding_bars} bars) "
                f"for {symbol} | Closing position at market price: {close_price:.2f}"
            )
            return self.broker.close_active_position(
                symbol=symbol,
                exit_price=close_price,
                timestamp=timestamp
            )
        return None

    def _create_and_execute_order(
        self,
        signal: Signal,
        symbol: str,
        timestamp: float,
        current_atr: Optional[Decimal] = None,
        adaptive_risk_params: Optional[AdaptiveRiskParams] = None
    ) -> Order:
        """
        Helper method to create and send an order to the broker.
        """
        # Calculate order quantity using Decimal
        close_dec = Decimal(str(signal.candle_close))
        trade_qty_usdt_dec = Decimal(str(TRADING.trade_quantity_usdt))
        quantity = trade_qty_usdt_dec / close_dec

        # Deterministic and traceable order ID (idempotency key)
        # Unique to timestamp, symbol, and signal type direction
        raw_payload = f"{timestamp}-{symbol}-{signal.type.name}"
        order_id = f"ord-{hashlib.sha256(raw_payload.encode()).hexdigest()[:12]}"

        # Calculate ATR-based Stop Loss and Take Profit prices if enabled
        sl_price = None
        tp_price = None
        atr_at_entry = current_atr

        if TRADING.use_atr_risk_management:
            if adaptive_risk_params is not None:
                atr_at_entry = Decimal(str(adaptive_risk_params.calculated_atr))
                sl_price = Decimal(str(adaptive_risk_params.sl_price))
                tp_price = Decimal(str(adaptive_risk_params.tp_price))
            elif current_atr is not None:
                sl_multiplier = Decimal(str(TRADING.atr_sl_multiplier))
                tp_multiplier = Decimal(str(TRADING.atr_tp_multiplier))
                tick_size = self._get_tick_size(symbol)

                if signal.type == OrderType.LONG:
                    sl_raw = close_dec - (sl_multiplier * current_atr)
                    tp_raw = close_dec + (tp_multiplier * current_atr)
                else:  # SHORT
                    sl_raw = close_dec + (sl_multiplier * current_atr)
                    tp_raw = close_dec - (tp_multiplier * current_atr)

                # Round prices according to Binance exchange tickSize filter
                sl_price = sl_raw.quantize(tick_size)
                tp_price = tp_raw.quantize(tick_size)
            else:
                logger.warning(
                    f"[OrderManager] ATR risk management is enabled, but neither current_atr nor adaptive_risk_params is provided. "
                    f"Order will be placed without ATR protection."
                )

        order = Order(
            order_id=order_id,
            symbol=symbol,
            type=signal.type,
            quantity=quantity,
            price=close_dec,
            status=OrderStatus.PENDING,
            timestamp=timestamp,
            signal=signal,
            atr_at_entry=atr_at_entry,
            sl_price=sl_price,
            tp_price=tp_price
        )

        logger.info(
            f"[OrderManager] Placing {order.type.name} entry order for {symbol} | "
            f"Qty: {float(order.quantity):.6f} | Price: {float(order.price):.2f} | "
            f"ATR: {float(atr_at_entry) if atr_at_entry is not None else 'N/A'} | "
            f"SL: {float(sl_price) if sl_price is not None else 'N/A'} | "
            f"TP: {float(tp_price) if tp_price is not None else 'N/A'}"
        )
        try:
            self.broker.execute_order(order)
        except ValueError as e:
            logger.error(f"[OrderManager] Failed to execute order: {e}")
        return order