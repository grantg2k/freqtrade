"""Alpaca exchange subclass using Alpaca's official SDK."""

from __future__ import annotations

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


# import alpaca_trade_api as tradeapi


logger = logging.getLogger(__name__)


class Alpaca(Exchange):
    """Alpaca exchange class implementing basic endpoints."""

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,
        "always_require_api_keys": True,
    }

    # def set_markets_from_exchange(self) -> None:
    #     """Set markets from exchange."""
    #
    # def additional_exchange_init(self) -> None:
    #     """Set API domain and initialize Alpaca clients."""
    #     paper = self._config.get("paper", True)
    #     base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    #     try:
    #         if hasattr(self._api, "urls"):
    #             self._api.urls.setdefault("api", base_url)
    #             self._api.urls["api"] = base_url
    #     except Exception as e:  # pragma: no cover - defensive
    #         logger.warning("Failed to set Alpaca base url: %s", e)
    #     key = self._config.get("key") or self._config.get("api_key")
    #     secret = self._config.get("secret")
    #     self._rest = tradeapi.REST(key, secret, base_url)
    #     self._async = tradeapi.AsyncRest(key, secret, base_url)
    # # ------------------------------------------------------------------
    # # Helper conversion methods
    # # ------------------------------------------------------------------
    # @staticmethod
    # def _to_symbol(pair: str) -> str:
    #     return pair.replace("/", "")
    # @staticmethod
    # def _tf(timeframe: str) -> tradeapi.rest.TimeFrame:
    #     from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
    #     mapping = {
    #         "1m": TimeFrame.Minute,
    #         "5m": TimeFrame(5, TimeFrameUnit.Minute),
    #         "15m": TimeFrame(15, TimeFrameUnit.Minute),
    #         "30m": TimeFrame(30, TimeFrameUnit.Minute),
    #         "1h": TimeFrame.Hour,
    #         "1d": TimeFrame.Day,
    #     }
    #     return mapping.get(timeframe, TimeFrame.Minute)

    # async def _async_call(self, func, *args, **kwargs):
    #     loop = asyncio.get_running_loop()
    #     return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    # # ------------------------------------------------------------------
    # # Minimal ccxt compatible methods
    # # ------------------------------------------------------------------
    # def fetch_markets(self) -> list[dict[str, Any]]:
    #     assets = self.loop.run_until_complete(self._async_call(self._rest.list_assets))
    #     return [{"symbol": a.symbol, "name": a.name, "status": a.status} for a in assets]
    # def fetch_balance(self) -> dict[str, Any]:
    #     account = self.loop.run_until_complete(self._async_call(self._rest.get_account))
    #     return {
    #         account.currency: {
    #             "free": float(account.cash),
    #             "total": float(account.cash),
    #         }
    #     }
    # def fetch_ohlcv(
    #     self,
    #     pair: str,
    #     timeframe: str = "1m",
    #     since: Optional[int] = None,
    #     limit: int = 1000,
    # ) -> list[list[Any]]:
    #     symbol = self._to_symbol(pair)
    #     tf = self._tf(timeframe)
    #     start = datetime.utcfromtimestamp(since / 1000).isoformat() if since else None
    #     bars = self.loop.run_until_complete(
    #         self._async.get_crypto_bars(symbol, tf, start=start, limit=limit)
    #     )
    #     return [
    #         [int(bar.t.timestamp() * 1000), bar.o, bar.h, bar.l, bar.c, bar.v]
    #         for bar in bars
    #     ]
    # def create_order(
    #     self,
    #     pair: str,
    #     side: str,
    #     amount: float,
    #     price: Optional[float] = None,
    # ) -> dict[str, Any]:
    #     symbol = self._to_symbol(pair)
    #     order = self.loop.run_until_complete(
    #         self._async_call(
    #             self._rest.submit_order,
    #             symbol,
    #             qty=amount,
    #             side=side,
    #             type=type,
    #             time_in_force="gtc",
    #             limit_price=price,
    #         )
    #     )
    #     return order._raw
    # def fetch_order(self, order_id: str) -> dict[str, Any]:
    #     order = self.loop.run_until_complete(self._async_call(self._rest.get_order, order_id))
    #     return order._raw
    # def fetch_open_orders(self, pair: Optional[str] = None) -> Iterable[dict[str, Any]]:
    #     orders = self.loop.run_until_complete(
    #         self._async_call(self._rest.list_orders, status="open",symbols=[
    # self._to_symbol(pair)] if pair else None)
    #     )
    #     return [o._raw for o in orders]
    # def fetch_closed_orders(self, pair: Optional[str] = None) ->
    # Iterable[dict[str, Any]]:
    #     orders = self.loop.run_until_complete(
    #         self._async_call(self._rest.list_orders, status="closed",symbols=[
    # self._to_symbol(pair)] if pair else None)
    #     )
    #     return [o._raw for o in orders]
    # def cancel_order(self, order_id: str) -> Any:
    #     return self.loop.run_until_complete(self._async_call(self._rest.cancel_order, order_id))
    # def fetch_ticker(self, pair: str) -> dict[str, Any]:
    #     symbol = self._to_symbol(pair)
    #     trade = self.loop.run_until_complete(self._async.get_latest_crypto_trade(symbol))
    #     return {"symbol": pair, "price": trade.p, "timestamp": int(trade.t.timestamp() * 1000)}
    # def fetch_trades(self, pair: str, limit: int = 100) -> Iterable[dict[str, Any]]:
    #     symbol = self._to_symbol(pair)
    #     trades = self.loop.run_until_complete(
    #         self._async.get_crypto_trades(symbol, limit=limit)
    #     )
    #     return [t._raw for t in trades]
