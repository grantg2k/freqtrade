"""BTCC exchange subclass."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from urllib.parse import urlencode

import websockets

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

    This class integrates the BTCC WebSocket API with Freqtrade based on the
    public documentation.  Communication is handled via WebSocket messages using
    actions such as ``GetActiveContracts`` or ``PlaceOrder``.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 500,
        "ws_enabled": True,
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit"},
        "order_time_in_force": ["DAY", "GTC"],
    }

    # WebSocket endpoint for BTCC
    ws_url: str = "wss://ws.btcc.com"

    def _sign_payload(self, payload: dict) -> dict:
        """Add common authentication fields and sign the payload."""
        payload = payload.copy()
        payload.update(
            {
                "timestamp": int(time.time() * 1000),
                "nonce": str(secrets.randbelow(90_000_000) + 10_000_000),
                "public_key": self._config["exchange"].get("key", ""),
                "crid": str(uuid.uuid4()),
            }
        )
        query = urlencode(dict(sorted(payload.items())))
        secret = self._config["exchange"].get("secret", "")
        payload["sign"] = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return payload

    async def _ws_call(self, data: dict) -> dict:
        """Send data via WebSocket and wait for the response."""
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps(data))
            msg = await ws.recv()
        return json.loads(msg)

    def _request(self, action: str, params: dict | None = None) -> dict:
        """Internal helper to perform a single WebSocket request."""
        payload = {"action": action}
        if params:
            payload.update(params)
        signed = self._sign_payload(payload)
        return self.loop.run_until_complete(self._ws_call(signed))

    def fetch_markets(self, params: dict | None = None):
        """Retrieve markets using BTCC's ``GetActiveContracts`` endpoint."""
        data = self._request("GetActiveContracts", params)
        return data.get("Contracts", [])

    def fetch_trades(
        self,
        pair: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        """Fetch trades using BTCC's ``GetTrades`` endpoint."""
        req = {
            "symbol": pair.replace("/", "_") if pair else None,
            "count": limit or self._ft_has.get("trades_limit", 100),
        }
        if params:
            req.update(params)
        data = self._request("GetTrades", req)
        return data.get("Trades", [])

    def fetch_balance(self, params: dict | None = None) -> CcxtBalances:
        """Return account balances via ``GetAccountInfo``."""
        data = self._request("GetAccountInfo", params)
        return data.get("AccountInfo", {})

    def fetch_ticker(self, pair: str, params: dict | None = None) -> Ticker:
        """Retrieve ticker information via ``Quote``."""
        req = {"symbol": pair.replace("/", "_")}
        data = self._request("Quote", req)
        return data.get("Ticker", {})

    def fetch_tickers(self, pairs: list[str] | None = None, params: dict | None = None) -> Tickers:
        """Retrieve multiple tickers using the ``SubscribeAllTickers`` action."""
        data = self._request("SubscribeAllTickers", params)
        return data.get("data", [])

    def fetch_order_book(
        self, pair: str, limit: int | None = None, params: dict | None = None
    ) -> OrderBook:
        """Retrieve orderbook using the ``Subscribe`` action."""
        req = {"symbol": pair.replace("/", "_")}
        data = self._request("Subscribe", req)
        return data.get("OrderBook", {})

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ):
        """Return OHLCV candles via ``GetKLine``."""
        req = {
            "symbol": pair.replace("/", "_"),
            "type": timeframe,
            "count": limit or self._ft_has.get("ohlcv_candle_limit", 500),
        }
        if since:
            req["since"] = since
        if params:
            req.update(params)
        data = self._request("GetKLine", req)
        return data.get("KLine", [])

    def create_order(
        self,
        *,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
    ) -> CcxtOrder:
        """Place an order via BTCC's ``PlaceOrder`` endpoint."""
        req = {
            "symbol": pair.replace("/", "_"),
            "side": side.upper(),
            "order_type": ordertype.upper(),
            "price": rate,
            "quantity": amount,
        }
        if reduceOnly:
            req["reduceOnly"] = True
        if time_in_force:
            req["time_in_force"] = time_in_force
        data = self._request("PlaceOrder", req)
        return {"info": data, "id": data.get("OID")}

    def cancel_order(
        self, order_id: str, pair: str | None = None, params: dict | None = None
    ) -> CcxtOrder:
        """Cancel an order using the ``CancelOrder`` endpoint."""
        req = {"symbol": pair.replace("/", "_") if pair else None, "order_id": order_id}
        if params:
            req.update(params)
        data = self._request("CancelOrder", req)
        return {"info": data, "id": data.get("OID")}

    def fetch_order(
        self, order_id: str, pair: str | None = None, params: dict | None = None
    ) -> CcxtOrder:
        """Fetch a single order via ``RetrieveOrder``."""
        req = {"symbol": pair.replace("/", "_") if pair else None, "order_id": order_id}
        if params:
            req.update(params)
        data = self._request("RetrieveOrder", req)
        return {"info": data, "id": order_id}

    def fetch_open_orders(
        self,
        pair: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[CcxtOrder]:
        """Retrieve open orders from ``GetOpenOrders``."""
        req = {"symbol": pair.replace("/", "_") if pair else None}
        if params:
            req.update(params)
        data = self._request("GetOpenOrders", req)
        return data.get("Orders", [])

    def fetch_closed_orders(
        self,
        pair: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[CcxtOrder]:
        """Retrieve closed orders from ``GetClosedOrders``."""
        req = {"symbol": pair.replace("/", "_") if pair else None}
        if params:
            req.update(params)
        data = self._request("GetClosedOrders", req)
        return data.get("Orders", [])

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
        """Place a stop-loss order using the ``PlaceOrder`` endpoint."""
        req = {
            "symbol": pair.replace("/", "_"),
            "side": side.upper(),
            "order_type": ordertype.upper(),
            "price": price,
            "stop_price": stop_price,
            "quantity": amount,
        }
        if params:
            req.update(params)
        data = self._request("PlaceOrder", req)
        return {"info": data, "id": data.get("OID")}
