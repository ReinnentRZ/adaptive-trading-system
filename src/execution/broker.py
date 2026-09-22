from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional
from src.models.order import Order
from src.models.position import Position
from src.models.trade import Trade


class Broker(ABC):
    """
    Abstract Base Class (Interface) representing a Broker/Exchange connection.
    Defines methods for execution, position tracking, and account querying.
    All numeric/financial values utilize Decimal for precision.
    """
    @abstractmethod
    def execute_order(self, order: Order) -> Trade:
        """
        Executes a given Order (buy or sell) on the exchange.
        Returns the resulting Trade execution.
        """
        pass

    @abstractmethod
    def get_active_position(self, symbol: str) -> Optional[Position]:
        """
        Retrieves the active open Position for a given symbol, if it exists.
        """
        pass

    @abstractmethod
    def close_active_position(
        self,
        symbol: str,
        exit_price: float | Decimal,
        timestamp: float
    ) -> Optional[Trade]:
        """
        Closes the active Position for a given symbol at the specified exit price.
        Returns the resulting Trade execution of the exit.
        """
        pass

    @abstractmethod
    def get_balance(self) -> Decimal:
        """
        Retrieves the current cash balance of the trading account.
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Retrieves trading statistics (Total Trades, Win Rate, PnL, etc.).
        """
        pass

    @abstractmethod
    def increment_position_bars(self, symbol: str) -> Optional[Position]:
        """
        Increments the bars held counter for an active position.
        Returns the updated Position or None if no active position exists.
        """
        pass

