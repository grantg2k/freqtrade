"""BTCC exchange subclass (stub)."""

import logging

from freqtrade.constants import BuySell
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import (
    CcxtBalances,
    CcxtOrder,
    FtHas,
    OrderBook,
    Ticker,
    Tickers,
)


logger = logging.getLogger(__name__)


class Btcc(Exchange):
    """BTCC exchange class.

    This class integrates the BTCC API with Freqtrade.  The implementation is a
    placeholder based on the public documentation.  Endpoint logic must be added
    to communicate with BTCC's API (GetActiveContracts, GetTrades, PlaceOrder,
    etc.).
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 500,
        "ws_enabled": True,
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit"},
    }

    # Base REST endpoint for BTCC
    base_url: str = "https://api.btcc.com/api/v3"

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Internal helper for REST requests to BTCC."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_markets(self, params: dict | None = None):
        """Retrieve markets using BTCC's GetActiveContracts endpoint."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_trades(
        self,
        pair: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        """Fetch trades using BTCC's GetTrades endpoint."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_balance(self, params: dict | None = None) -> CcxtBalances:
        """Return account balances via GetAccountBalance."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_ticker(self, pair: str, params: dict | None = None) -> Ticker:
        """Retrieve ticker information via GetTicker."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_tickers(self, pairs: list[str] | None = None, params: dict | None = None) -> Tickers:
        """Retrieve multiple tickers using the GetTickers endpoint."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_order_book(
        self, pair: str, limit: int | None = None, params: dict | None = None
    ) -> OrderBook:
        """Retrieve orderbook using GetMarketDepth."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        """Return OHLCV candles via GetKLine."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def create_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ):
        """Place an order via BTCC's PlaceOrder endpoint."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def cancel_order(
        self, order_id: str, pair: str | None = None, params: dict | None = None
    ) -> CcxtOrder:
        """Cancel an order using the CancelOrder endpoint."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_order(
        self, order_id: str, pair: str | None = None, params: dict | None = None
    ) -> CcxtOrder:
        """Fetch a single order via GetOrder."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_open_orders(
        self,
        pair: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[CcxtOrder]:
        """Retrieve open orders from GetOrders."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def fetch_closed_orders(
        self,
        pair: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[CcxtOrder]:
        """Retrieve closed orders from GetOrdersHistory."""
        raise NotImplementedError("BTCC API integration not implemented.")

    def create_stoploss_order(
        self,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        price: float,
        stop_price: float,
        params: dict | None = None,
    ) -> CcxtOrder:
        """Place a stop-loss order using the PlacePlan endpoint."""
        raise NotImplementedError("BTCC API integration not implemented.")

