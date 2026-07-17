"""Alpaca exchange implementation using the official :mod:`alpaca-py` SDK.

Alpaca is not implemented by CCXT.  The small adapters in this module expose
the subset of CCXT's unified API consumed by :class:`Exchange`, keeping the
rest of Freqtrade's exchange lifecycle, dry-run support and error handling
unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import ccxt
from ccxt import TICK_SIZE

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


def _value(value: Any) -> Any:
    """Return the scalar value of SDK enums while leaving other values alone."""
    return getattr(value, "value", value)


def _float(value: Any, default: float = 0.0) -> float:
    return default if value in (None, "") else float(value)


def _timestamp(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def _sdk() -> SimpleNamespace:
    """Import alpaca-py lazily so importing Freqtrade remains lightweight."""
    try:
        trading_client = import_module("alpaca.trading.client")
        trading_enums = import_module("alpaca.trading.enums")
        trading_requests = import_module("alpaca.trading.requests")
        common_enums = import_module("alpaca.common.enums")
        historical = import_module("alpaca.data.historical")
        data_requests = import_module("alpaca.data.requests")
        timeframe = import_module("alpaca.data.timeframe")
    except ImportError as exc:
        raise OperationalException(
            "Alpaca support requires the official 'alpaca-py' package. "
            "Install Freqtrade's requirements before selecting this exchange."
        ) from exc

    return SimpleNamespace(
        TradingClient=trading_client.TradingClient,
        StockHistoricalDataClient=historical.StockHistoricalDataClient,
        CryptoHistoricalDataClient=historical.CryptoHistoricalDataClient,
        OptionHistoricalDataClient=historical.OptionHistoricalDataClient,
        TimeFrame=timeframe.TimeFrame,
        TimeFrameUnit=timeframe.TimeFrameUnit,
        Sort=common_enums.Sort,
        **{
            name: getattr(trading_enums, name)
            for name in (
                "AssetClass",
                "AssetStatus",
                "OrderSide",
                "QueryOrderStatus",
                "TimeInForce",
            )
            if hasattr(trading_enums, name)
        },
        **{
            name: getattr(trading_requests, name)
            for name in (
                "GetAssetsRequest",
                "GetOptionContractsRequest",
                "GetOrdersRequest",
                "LimitOrderRequest",
                "MarketOrderRequest",
                "StopLimitOrderRequest",
                "StopOrderRequest",
            )
        },
        **{
            name: getattr(data_requests, name)
            for name in (
                "CryptoBarsRequest",
                "CryptoLatestOrderbookRequest",
                "CryptoLatestQuoteRequest",
                "CryptoLatestTradeRequest",
                "CryptoTradesRequest",
                "OptionBarsRequest",
                "OptionLatestQuoteRequest",
                "OptionLatestTradeRequest",
                "OptionTradesRequest",
                "StockBarsRequest",
                "StockLatestQuoteRequest",
                "StockLatestTradeRequest",
                "StockTradesRequest",
            )
        },
    )


class _AlpacaApi:
    """Synchronous CCXT-shaped facade around alpaca-py clients."""

    id = "alpaca"
    name = "Alpaca"
    precisionMode = TICK_SIZE
    timeframes = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "1h": "1Hour",
        "1d": "1Day",
        "1w": "1Week",
        "1M": "1Month",
    }
    has = {
        "cancelOrder": True,
        "createOrder": True,
        "createLimitOrder": True,
        "createMarketOrder": True,
        "fetchBalance": True,
        "fetchBidsAsks": True,
        "fetchClosedOrders": True,
        "fetchL2OrderBook": True,
        "fetchMyTrades": True,
        "fetchOHLCV": True,
        "fetchOpenOrders": True,
        "fetchOrder": True,
        "fetchOrders": True,
        "fetchTicker": True,
        "fetchTickers": True,
        "fetchTrades": True,
        "watchOHLCV": False,
    }
    features = {"spot": {"fetchOHLCV": {"limit": 10000}}}

    def __init__(self, exchange_config: dict[str, Any]) -> None:
        self.sdk = _sdk()
        self.options = {"createMarketBuyOrderRequiresPrice": False}
        self.markets: dict[str, dict[str, Any]] = {}
        self.currencies: dict[str, dict[str, Any]] = {}
        self.session = None

        key = exchange_config.get(
            "api_key", exchange_config.get("apiKey", exchange_config.get("key"))
        )
        secret = exchange_config.get("secret")
        paper = exchange_config.get("paper", exchange_config.get("sandbox", True))
        trading_url = exchange_config.get("trading_url")
        data_url = exchange_config.get("data_url")

        trading_kwargs = {"paper": paper}
        if trading_url:
            trading_kwargs["url_override"] = trading_url
        data_kwargs = {"url_override": data_url} if data_url else {}

        self.trading = self.sdk.TradingClient(key, secret, **trading_kwargs)
        self.stock_data = self.sdk.StockHistoricalDataClient(key, secret, **data_kwargs)
        self.crypto_data = self.sdk.CryptoHistoricalDataClient(key, secret, **data_kwargs)
        self.option_data = self.sdk.OptionHistoricalDataClient(key, secret, **data_kwargs)
        self.include_options = exchange_config.get("include_options", True)
        self.option_contract_limit = exchange_config.get("option_contract_limit", 1000)
        self.option_underlyings = exchange_config.get("option_underlyings")

    @staticmethod
    def _raise_sdk_error(exc: Exception) -> None:
        message = str(exc)
        status = getattr(exc, "status_code", getattr(exc, "status", None))
        lowered = message.lower()
        if status == 429:
            raise ccxt.DDoSProtection(message) from exc
        if status in (401, 403):
            raise ccxt.AuthenticationError(message) from exc
        if status == 404:
            raise ccxt.OrderNotFound(message) from exc
        if "insufficient" in lowered or "buying power" in lowered:
            raise ccxt.InsufficientFunds(message) from exc
        if status in (400, 422) or "invalid" in lowered:
            raise ccxt.InvalidOrder(message) from exc
        raise ccxt.ExchangeError(message) from exc

    def _call(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except ccxt.BaseError:
            raise
        except Exception as exc:
            self._raise_sdk_error(exc)

    @staticmethod
    def _market_symbol(asset_id: str, asset_class: str) -> str:
        if asset_class == "crypto":
            return asset_id
        return f"{asset_id}/USD"

    def _market(self, item: Any, asset_class: str) -> dict[str, Any]:
        item_id = str(item.symbol)
        symbol = self._market_symbol(item_id, asset_class)
        quote = item_id.split("/", 1)[1] if asset_class == "crypto" else "USD"
        base = item_id.split("/", 1)[0] if asset_class == "crypto" else item_id
        amount_step = getattr(item, "min_trade_increment", None)
        price_step = getattr(item, "price_increment", None)
        if asset_class == "option":
            amount_step = 1.0
            price_step = 0.01
        return {
            "id": item_id,
            "symbol": symbol,
            "base": base,
            "quote": quote,
            "settle": None,
            "baseId": item_id,
            "quoteId": quote,
            "type": "spot",
            "spot": True,
            "margin": False,
            "swap": False,
            "future": False,
            "option": asset_class == "option",
            "active": _value(getattr(item, "status", "active")) == "active"
            and bool(getattr(item, "tradable", True)),
            "contract": asset_class == "option",
            "contractSize": _float(getattr(item, "size", None), 1.0),
            "precision": {"amount": amount_step or 1e-9, "price": price_step or 0.01},
            "limits": {
                "amount": {
                    "min": getattr(item, "min_order_size", None) or amount_step,
                    "max": None,
                },
                "price": {"min": price_step, "max": None},
                "cost": {"min": None, "max": None},
                "leverage": {"min": None, "max": None},
            },
            "info": item,
            "alpaca_asset_class": asset_class,
        }

    def fetch_markets(self, params: dict | None = None) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        for asset_class in ("us_equity", "crypto"):
            request = self.sdk.GetAssetsRequest(
                status=self.sdk.AssetStatus.ACTIVE,
                asset_class=getattr(self.sdk.AssetClass, asset_class.upper()),
            )
            assets = self._call(self.trading.get_all_assets, request)
            markets.extend(
                self._market(asset, "crypto" if asset_class == "crypto" else "equity")
                for asset in assets
                if getattr(asset, "tradable", False)
            )

        if self.include_options:
            token = None
            remaining = int(self.option_contract_limit)
            while remaining > 0:
                request = self.sdk.GetOptionContractsRequest(
                    status=self.sdk.AssetStatus.ACTIVE,
                    underlying_symbols=self.option_underlyings,
                    limit=min(remaining, 10000),
                    page_token=token,
                )
                response = self._call(self.trading.get_option_contracts, request)
                contracts = getattr(response, "option_contracts", None) or []
                markets.extend(
                    self._market(contract, "option")
                    for contract in contracts
                    if getattr(contract, "tradable", False)
                )
                remaining -= len(contracts)
                token = getattr(response, "next_page_token", None)
                if not token or not contracts:
                    break
        return markets

    def load_markets(self, reload: bool = False, params: dict | None = None) -> dict[str, Any]:
        if reload or not self.markets:
            self.markets = {market["symbol"]: market for market in self.fetch_markets(params)}
            codes = {market["base"] for market in self.markets.values()} | {
                market["quote"] for market in self.markets.values()
            }
            self.currencies = {code: {"id": code, "code": code} for code in codes}
        return self.markets

    def set_markets_from_exchange(self, other: Any) -> None:
        self.markets = other.markets.copy()
        self.currencies = other.currencies.copy()

    def market(self, symbol: str) -> dict[str, Any]:
        try:
            return self.markets[symbol]
        except KeyError as exc:
            raise ccxt.BadSymbol(f"Unknown Alpaca market {symbol}") from exc

    def _asset_id(self, symbol: str) -> str:
        return self.market(symbol)["id"]

    def fetch_balance(self, params: dict | None = None) -> dict[str, Any]:
        account = self._call(self.trading.get_account)
        positions = self._call(self.trading.get_all_positions)
        cash = _float(getattr(account, "cash", None))
        result: dict[str, Any] = {
            "free": {"USD": cash},
            "used": {"USD": 0.0},
            "total": {"USD": cash},
            "USD": {"free": cash, "used": 0.0, "total": cash},
            "info": account,
        }
        for position in positions:
            code = str(position.symbol).split("/", 1)[0]
            total = abs(_float(position.qty))
            free = abs(_float(getattr(position, "qty_available", None), total))
            balance = {"free": free, "used": max(total - free, 0.0), "total": total}
            result[code] = balance
            for field in ("free", "used", "total"):
                result[field][code] = balance[field]
        return result

    def calculate_fee(
        self,
        symbol: str,
        type: str,  # noqa: A002 - CCXT-compatible keyword name
        side: str,
        amount: float,
        price: float,
        takerOrMaker: str = "taker",
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Return Alpaca's commission rate; regulatory fees arrive on actual fills."""
        return {
            "type": takerOrMaker,
            "currency": self.market(symbol)["quote"],
            "rate": 0.0,
            "cost": 0.0,
        }

    def _timeframe(self, timeframe: str) -> Any:
        unit = self.sdk.TimeFrameUnit
        if timeframe.endswith("m"):
            return self.sdk.TimeFrame(int(timeframe[:-1]), unit.Minute)
        if timeframe.endswith("h"):
            return self.sdk.TimeFrame(int(timeframe[:-1]), unit.Hour)
        if timeframe == "1d":
            return self.sdk.TimeFrame.Day
        if timeframe == "1w":
            return self.sdk.TimeFrame.Week
        if timeframe == "1M":
            return self.sdk.TimeFrame.Month
        raise ccxt.BadRequest(f"Unsupported Alpaca timeframe: {timeframe}")

    def _data_route(self, symbol: str, kind: str) -> tuple[Any, type, Callable]:
        market = self.market(symbol)
        asset_id = market["id"]
        asset_class = market["alpaca_asset_class"]
        title = {"equity": "Stock", "crypto": "Crypto", "option": "Option"}[asset_class]
        client = {
            "equity": self.stock_data,
            "crypto": self.crypto_data,
            "option": self.option_data,
        }[asset_class]
        request_class = getattr(self.sdk, f"{title}{kind}Request")
        method = getattr(client, f"get_{title.lower()}_{kind.lower()}")
        return asset_id, request_class, method

    @staticmethod
    def _data_values(response: Any, asset_id: str) -> list[Any]:
        data = getattr(response, "data", response)
        if isinstance(data, dict):
            return data.get(asset_id, [])
        return list(data or [])

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[list[float]]:
        asset_id, request_class, method = self._data_route(symbol, "Bars")
        request = request_class(
            symbol_or_symbols=asset_id,
            timeframe=self._timeframe(timeframe),
            start=datetime.fromtimestamp(since / 1000, UTC) if since is not None else None,
            limit=min(limit or 1000, 10000),
        )
        bars = self._data_values(self._call(method, request), asset_id)
        return [
            [
                _timestamp(bar.timestamp),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
            ]
            for bar in bars
        ]

    @staticmethod
    def _order_status(status: Any) -> str:
        status = _value(status)
        if status == "filled":
            return "closed"
        if status in {"canceled", "expired", "rejected", "replaced", "stopped"}:
            return "canceled"
        return "open"

    def _order(self, order: Any) -> dict[str, Any]:
        amount = _float(getattr(order, "qty", None))
        filled = _float(getattr(order, "filled_qty", None))
        average = getattr(order, "filled_avg_price", None)
        price = getattr(order, "limit_price", None) or average
        asset_class = _value(getattr(order, "asset_class", "us_equity"))
        raw_symbol = str(getattr(order, "symbol", ""))
        symbol = self._market_symbol(raw_symbol, asset_class)
        return {
            "id": str(order.id),
            "clientOrderId": getattr(order, "client_order_id", None),
            "timestamp": _timestamp(getattr(order, "submitted_at", None)),
            "datetime": getattr(order, "submitted_at", None).isoformat()
            if getattr(order, "submitted_at", None)
            else None,
            "lastTradeTimestamp": _timestamp(getattr(order, "filled_at", None)),
            "symbol": symbol,
            "type": _value(getattr(order, "type", None)),
            "timeInForce": str(_value(getattr(order, "time_in_force", "gtc"))).upper(),
            "side": _value(getattr(order, "side", None)),
            "price": _float(price) if price is not None else None,
            "average": _float(average) if average is not None else None,
            "amount": amount,
            "filled": filled,
            "remaining": max(amount - filled, 0.0),
            "cost": filled * _float(average),
            "status": self._order_status(order.status),
            "fee": {"cost": 0.0, "currency": "USD"},
            "trades": [],
            "info": order,
        }

    def create_order(
        self,
        symbol: str,
        type: str,  # noqa: A002 - CCXT-compatible keyword name
        side: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        common = {
            "symbol": self._asset_id(symbol),
            "qty": amount,
            "side": self.sdk.OrderSide(side),
            "time_in_force": self.sdk.TimeInForce(params.get("timeInForce", "gtc").lower()),
        }
        if type == "market":
            request = self.sdk.MarketOrderRequest(**common)
        elif type == "limit":
            request = self.sdk.LimitOrderRequest(**common, limit_price=price)
        elif type == "stop":
            request = self.sdk.StopOrderRequest(
                **common, stop_price=params.get("stopLossPrice", params.get("stopPrice", price))
            )
        elif type == "stop_limit":
            request = self.sdk.StopLimitOrderRequest(
                **common,
                limit_price=price,
                stop_price=params.get("stopLossPrice", params.get("stopPrice")),
            )
        else:
            raise ccxt.InvalidOrder(f"Unsupported Alpaca order type: {type}")
        return self._order(self._call(self.trading.submit_order, request))

    def fetch_order(
        self, order_id: str, symbol: str | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        return self._order(self._call(self.trading.get_order_by_id, order_id))

    def _fetch_orders(
        self,
        status: Any,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        request = self.sdk.GetOrdersRequest(
            status=status,
            symbols=[self._asset_id(symbol)] if symbol else None,
            after=datetime.fromtimestamp(since / 1000, UTC) if since is not None else None,
            limit=min(limit or 500, 500),
            direction=self.sdk.Sort.DESC,
            nested=True,
        )
        return [self._order(order) for order in self._call(self.trading.get_orders, request)]

    def fetch_open_orders(
        self, symbol: str | None = None, since: int | None = None, limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        return self._fetch_orders(self.sdk.QueryOrderStatus.OPEN, symbol, since, limit)

    def fetch_closed_orders(
        self, symbol: str | None = None, since: int | None = None, limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        return self._fetch_orders(self.sdk.QueryOrderStatus.CLOSED, symbol, since, limit)

    def fetch_orders(
        self, symbol: str | None = None, since: int | None = None, limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        return self._fetch_orders(self.sdk.QueryOrderStatus.ALL, symbol, since, limit)

    def cancel_order(
        self, order_id: str, symbol: str | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        self._call(self.trading.cancel_order_by_id, order_id)
        try:
            order = self.fetch_order(order_id, symbol, params)
            # Alpaca's cancel endpoint returns before the order-state read is
            # guaranteed to converge.  The accepted cancellation is definitive.
            if order["status"] == "open":
                order["status"] = "canceled"
            return order
        except ccxt.OrderNotFound:
            return {"id": order_id, "symbol": symbol, "status": "canceled", "info": {}}

    def _latest(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[str]] = {"equity": [], "crypto": [], "option": []}
        symbol_by_id: dict[str, str] = {}
        for symbol in symbols:
            market = self.market(symbol)
            grouped[market["alpaca_asset_class"]].append(market["id"])
            symbol_by_id[market["id"]] = symbol

        result: dict[str, dict[str, Any]] = {}
        for asset_class, ids in grouped.items():
            if not ids:
                continue
            title = {"equity": "Stock", "crypto": "Crypto", "option": "Option"}[asset_class]
            client = {
                "equity": self.stock_data,
                "crypto": self.crypto_data,
                "option": self.option_data,
            }[asset_class]
            quote_request = getattr(self.sdk, f"{title}LatestQuoteRequest")
            trade_request = getattr(self.sdk, f"{title}LatestTradeRequest")
            # Keep multi-symbol URLs below common proxy/request-line limits.
            for offset in range(0, len(ids), 200):
                batch = ids[offset : offset + 200]
                quotes = self._call(
                    getattr(client, f"get_{title.lower()}_latest_quote"),
                    quote_request(symbol_or_symbols=batch),
                )
                trades = self._call(
                    getattr(client, f"get_{title.lower()}_latest_trade"),
                    trade_request(symbol_or_symbols=batch),
                )
                for item_id in batch:
                    quote = quotes.get(item_id)
                    trade = trades.get(item_id)
                    last = float(trade.price) if trade else None
                    result[symbol_by_id[item_id]] = {
                        "symbol": symbol_by_id[item_id],
                        "timestamp": _timestamp(getattr(trade or quote, "timestamp", None)),
                        "datetime": getattr(trade or quote, "timestamp", None).isoformat()
                        if getattr(trade or quote, "timestamp", None)
                        else None,
                        "bid": float(quote.bid_price) if quote else None,
                        "bidVolume": float(quote.bid_size) if quote else None,
                        "ask": float(quote.ask_price) if quote else None,
                        "askVolume": float(quote.ask_size) if quote else None,
                        "last": last,
                        "close": last,
                        "baseVolume": None,
                        "quoteVolume": None,
                        "percentage": None,
                        "info": {"quote": quote, "trade": trade},
                    }
        return result

    def fetch_ticker(self, symbol: str, params: dict | None = None) -> dict[str, Any]:
        return self._latest([symbol])[symbol]

    def fetch_tickers(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> dict[str, dict[str, Any]]:
        return self._latest(symbols or list(self.markets))

    def fetch_bids_asks(
        self, symbols: list[str] | None = None, params: dict | None = None
    ) -> dict[str, dict[str, Any]]:
        return self.fetch_tickers(symbols, params)

    def fetch_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        asset_id, request_class, method = self._data_route(symbol, "Trades")
        request = request_class(
            symbol_or_symbols=asset_id,
            start=datetime.fromtimestamp(since / 1000, UTC) if since is not None else None,
            limit=min(limit or 1000, 10000),
        )
        trades = self._data_values(self._call(method, request), asset_id)
        return [
            {
                "id": str(trade.id) if getattr(trade, "id", None) is not None else None,
                "timestamp": _timestamp(trade.timestamp),
                "datetime": trade.timestamp.isoformat(),
                "symbol": symbol,
                "side": None,
                "price": float(trade.price),
                "amount": float(trade.size),
                "cost": float(trade.price) * float(trade.size),
                "info": trade,
            }
            for trade in trades
        ]

    def fetch_my_trades(
        self,
        symbol: str | None = None,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        trades = []
        for order in self.fetch_closed_orders(symbol, since, limit, params):
            if order["filled"] <= 0 or order["average"] is None:
                continue
            trades.append(
                {
                    "id": order["id"],
                    "order": order["id"],
                    "timestamp": order["lastTradeTimestamp"] or order["timestamp"],
                    "datetime": order["datetime"],
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "price": order["average"],
                    "amount": order["filled"],
                    "cost": order["cost"],
                    "fee": order["fee"],
                    "info": order["info"],
                }
            )
        return trades

    def fetch_l2_order_book(
        self, symbol: str, limit: int | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        market = self.market(symbol)
        if market["alpaca_asset_class"] == "crypto":
            request = self.sdk.CryptoLatestOrderbookRequest(symbol_or_symbols=market["id"])
            books = self._call(self.crypto_data.get_crypto_latest_orderbook, request)
            book = books[market["id"]]
            return {
                "symbol": symbol,
                "bids": [[float(item.price), float(item.size)] for item in book.bids[:limit]],
                "asks": [[float(item.price), float(item.size)] for item in book.asks[:limit]],
                "timestamp": _timestamp(book.timestamp),
                "datetime": book.timestamp.isoformat(),
                "nonce": None,
            }
        ticker = self.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "bids": [[ticker["bid"], ticker["bidVolume"]]] if ticker["bid"] else [],
            "asks": [[ticker["ask"], ticker["askVolume"]]] if ticker["ask"] else [],
            "timestamp": ticker["timestamp"],
            "datetime": ticker["datetime"],
            "nonce": None,
        }


class _AlpacaAsyncApi(_AlpacaApi):
    """Async facade used by Freqtrade's candle/trade download machinery."""

    async def close(self) -> None:
        return None

    async def load_markets(
        self, reload: bool = False, params: dict | None = None
    ) -> dict[str, Any]:
        return await asyncio.to_thread(super().load_markets, reload, params)

    async def fetch_ohlcv(self, *args: Any, **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(super().fetch_ohlcv, *args, **kwargs)

    async def fetch_trades(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(super().fetch_trades, *args, **kwargs)


class Alpaca(Exchange):
    """Freqtrade exchange for Alpaca stocks, ETFs, options and crypto."""

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 10000,
        "always_require_api_keys": True,
        "order_time_in_force": ["GTC", "DAY", "IOC", "FOK", "OPG", "CLS"],
        "tickers_have_quoteVolume": False,
        "tickers_have_percentage": False,
        "trades_has_history": True,
        "trades_limit": 10000,
    }

    def _init_ccxt(
        self, exchange_config: dict[str, Any], sync: bool, ccxt_kwargs: dict[str, Any]
    ) -> _AlpacaApi:
        """Build a native Alpaca adapter in place of a CCXT exchange object."""
        return _AlpacaApi(exchange_config) if sync else _AlpacaAsyncApi(exchange_config)

    # These explicit helpers are useful to plugins and tests which use the
    # exchange interface directly instead of Freqtrade's higher-level methods.
    def fetch_markets(self, params: dict | None = None) -> list[dict[str, Any]]:
        return self._api.fetch_markets(params)

    def fetch_balance(self, params: dict | None = None) -> dict[str, Any]:
        return self._api.fetch_balance(params)

    def fetch_ohlcv(
        self, pair: str, timeframe: str = "1m", since: int | None = None,
        limit: int | None = None, params: dict | None = None,
    ) -> list[list[float]]:
        return self._api.fetch_ohlcv(pair, timeframe, since, limit, params)

    def fetch_open_orders(
        self, pair: str | None = None, since: int | None = None,
        limit: int | None = None, params: dict | None = None,
    ) -> list[dict[str, Any]]:
        return self._api.fetch_open_orders(pair, since, limit, params)

    def fetch_closed_orders(
        self, pair: str | None = None, since: int | None = None,
        limit: int | None = None, params: dict | None = None,
    ) -> list[dict[str, Any]]:
        return self._api.fetch_closed_orders(pair, since, limit, params)

    def fetch_tickers(
        self, pairs: list[str] | None = None, params: dict | None = None
    ) -> dict[str, dict[str, Any]]:
        return self._api.fetch_tickers(pairs, params)

    def fetch_trades(
        self, pair: str, since: int | None = None, limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        return self._api.fetch_trades(pair, since, limit, params)
