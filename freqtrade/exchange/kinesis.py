import hashlib
import hmac
import json
import logging
import time
from typing import Any

import requests

from freqtrade.constants import BuySell
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Kinesis(Exchange):
    """Kinesis exchange class using the public REST API."""

    _ft_has: FtHas = {
        "ohlcv_has_history": False,
        "trades_has_history": False,
        "ws_enabled": False,
    }

    def _sign(self, method: str, path: str, body: str = "") -> dict[str, str]:
        nonce = str(int(time.time() * 1000))
        message = f"{nonce}{method}{path}{body}".encode()
        secret = self._config["exchange"]["secret"].encode()
        signature = hmac.new(secret, message, hashlib.sha256).hexdigest().upper()
        headers = {
            "x-nonce": nonce,
            "x-api-key": self._config["exchange"]["key"],
            "x-signature": signature,
        }
        if method != "DELETE":
            headers["Content-Type"] = "application/json"
        return headers

    def _request(self, method: str, path: str, data: str | None = None) -> Any:
        base_url = self._config["exchange"].get("base_url", "https://client-api.kinesis.money")
        headers = self._sign(method, path, data or "")
        url = f"{base_url}{path}"
        response = requests.request(method, url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def fetch_balance(self):
        return self._request("GET", "/v1/exchange/holdings")

    def fetch_ticker(self, pair: str):
        return self._request("GET", f"/v1/exchange/mid-price/{pair}")

    def fetch_ohlcv(self, pair: str, timeframe: str, since=None, limit=None):
        return self._request("GET", f"/v1/exchange/ohlc/{pair}")

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
    ) -> dict[str, Any]:
        payload = {
            "currencyPairId": pair,
            "direction": side,
            "amount": float(amount),
            "orderType": ordertype,
            "limitPrice": rate,
        }
        return self._request(
            "POST", "/v1/exchange/orders", json.dumps(payload, separators=(",", ":"))
        )

    def cancel_order(
        self,
        order_id: str,
        pair: str,
        params: dict[Any, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/exchange/orders/{order_id}")

    def fetch_open_orders(self):
        return self._request("GET", "/v1/exchange/orders/open")
