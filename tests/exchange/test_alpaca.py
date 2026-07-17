from datetime import UTC, datetime
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from freqtrade.exchange.alpaca import _AlpacaApi, _AlpacaAsyncApi


class Request:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CallableEnum:
    def __call__(self, value):
        return value


class AssetClass(Enum):
    US_EQUITY = "us_equity"
    CRYPTO = "crypto"


def api() -> _AlpacaApi:
    instance = _AlpacaApi.__new__(_AlpacaApi)
    instance.options = {"createMarketBuyOrderRequiresPrice": False}
    instance.markets = {
        "AAPL/USD": {
            "id": "AAPL",
            "symbol": "AAPL/USD",
            "base": "AAPL",
            "quote": "USD",
            "alpaca_asset_class": "equity",
        },
        "BTC/USD": {
            "id": "BTC/USD",
            "symbol": "BTC/USD",
            "base": "BTC",
            "quote": "USD",
            "alpaca_asset_class": "crypto",
        },
        "AAPL260116C00200000/USD": {
            "id": "AAPL260116C00200000",
            "symbol": "AAPL260116C00200000/USD",
            "base": "AAPL260116C00200000",
            "quote": "USD",
            "alpaca_asset_class": "option",
        },
    }
    request_names = [
        "CryptoBarsRequest",
        "CryptoLatestOrderbookRequest",
        "CryptoLatestQuoteRequest",
        "CryptoLatestTradeRequest",
        "CryptoTradesRequest",
        "GetAssetsRequest",
        "GetOptionContractsRequest",
        "GetOrdersRequest",
        "LimitOrderRequest",
        "MarketOrderRequest",
        "OptionBarsRequest",
        "OptionLatestQuoteRequest",
        "OptionLatestTradeRequest",
        "OptionTradesRequest",
        "StockBarsRequest",
        "StockLatestQuoteRequest",
        "StockLatestTradeRequest",
        "StockTradesRequest",
        "StopLimitOrderRequest",
        "StopOrderRequest",
    ]
    sdk = {name: Request for name in request_names}
    sdk.update(
        {
            "AssetClass": AssetClass,
            "AssetStatus": SimpleNamespace(ACTIVE="active"),
            "OrderSide": CallableEnum(),
            "QueryOrderStatus": SimpleNamespace(OPEN="open", CLOSED="closed", ALL="all"),
            "Sort": SimpleNamespace(DESC="desc"),
            "TimeFrame": MagicMock(),
            "TimeFrameUnit": SimpleNamespace(Minute="minute", Hour="hour"),
            "TimeInForce": CallableEnum(),
        }
    )
    sdk["TimeFrame"].Day = "day"
    sdk["TimeFrame"].Week = "week"
    sdk["TimeFrame"].Month = "month"
    instance.sdk = SimpleNamespace(**sdk)
    instance.trading = MagicMock()
    instance.stock_data = MagicMock()
    instance.crypto_data = MagicMock()
    instance.option_data = MagicMock()
    instance.include_options = True
    instance.option_contract_limit = 100
    instance.option_underlyings = None
    instance.currencies = {}
    instance.session = None
    return instance


def test_fetch_markets_covers_equities_crypto_and_options() -> None:
    instance = api()
    equity = SimpleNamespace(
        symbol="SPY",
        status="active",
        tradable=True,
        min_trade_increment=0.001,
        min_order_size=0.001,
        price_increment=0.01,
    )
    crypto = SimpleNamespace(
        symbol="ETH/USD",
        status="active",
        tradable=True,
        min_trade_increment=1e-8,
        min_order_size=1e-5,
        price_increment=0.01,
    )
    option = SimpleNamespace(
        symbol="SPY260116C00600000",
        status="active",
        tradable=True,
        size="100",
    )
    instance.trading.get_all_assets.side_effect = [[equity], [crypto]]
    instance.trading.get_option_contracts.return_value = SimpleNamespace(
        option_contracts=[option], next_page_token=None
    )

    markets = {market["symbol"]: market for market in instance.fetch_markets()}

    assert set(markets) == {"SPY/USD", "ETH/USD", "SPY260116C00600000/USD"}
    assert markets["SPY/USD"]["spot"] is True
    assert markets["SPY260116C00600000/USD"]["option"] is True
    assert markets["SPY260116C00600000/USD"]["contractSize"] == 100.0


def test_fetch_balance_includes_cash_and_available_positions() -> None:
    instance = api()
    instance.trading.get_account.return_value = SimpleNamespace(cash="1250.50")
    instance.trading.get_all_positions.return_value = [
        SimpleNamespace(symbol="AAPL", qty="3", qty_available="2"),
        SimpleNamespace(symbol="BTC/USD", qty="0.5", qty_available="0.5"),
    ]

    balances = instance.fetch_balance()

    assert balances["USD"] == {"free": 1250.5, "used": 0.0, "total": 1250.5}
    assert balances["AAPL"] == {"free": 2.0, "used": 1.0, "total": 3.0}
    assert balances["BTC"]["total"] == 0.5


@pytest.mark.parametrize(
    "symbol,client,method",
    [
        ("AAPL/USD", "stock_data", "get_stock_bars"),
        ("BTC/USD", "crypto_data", "get_crypto_bars"),
        ("AAPL260116C00200000/USD", "option_data", "get_option_bars"),
    ],
)
def test_fetch_ohlcv_routes_each_asset_class(symbol: str, client: str, method: str) -> None:
    instance = api()
    now = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    bar = SimpleNamespace(timestamp=now, open=1, high=3, low=0.5, close=2, volume=10)
    getattr(getattr(instance, client), method).return_value = SimpleNamespace(
        data={instance.markets[symbol]["id"]: [bar]}
    )

    candles = instance.fetch_ohlcv(symbol, "5m", since=1_700_000_000_000, limit=20)

    assert candles == [[int(now.timestamp() * 1000), 1.0, 3.0, 0.5, 2.0, 10.0]]
    request = getattr(getattr(instance, client), method).call_args.args[0]
    assert request.limit == 20
    assert request.symbol_or_symbols == instance.markets[symbol]["id"]


def order(status: str = "new", asset_class: str = "us_equity") -> SimpleNamespace:
    submitted = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    return SimpleNamespace(
        id="order-1",
        client_order_id="client-1",
        submitted_at=submitted,
        filled_at=submitted if status == "filled" else None,
        symbol="BTC/USD" if asset_class == "crypto" else "AAPL",
        asset_class=asset_class,
        type="limit",
        time_in_force="gtc",
        side="buy",
        limit_price="100",
        filled_avg_price="99" if status == "filled" else None,
        qty="2",
        filled_qty="2" if status == "filled" else "0",
        status=status,
    )


def test_create_fetch_list_and_cancel_orders() -> None:
    instance = api()
    instance.trading.submit_order.return_value = order()
    instance.trading.get_order_by_id.return_value = order("filled")
    instance.trading.get_orders.return_value = [order("filled")]

    created = instance.create_order("AAPL/USD", "limit", "buy", 2, 100, {"timeInForce": "GTC"})
    fetched = instance.fetch_order("order-1", "AAPL/USD")
    closed = instance.fetch_closed_orders("AAPL/USD")
    canceled = instance.cancel_order("order-1", "AAPL/USD")

    assert created["status"] == "open"
    assert instance.trading.submit_order.call_args.args[0].limit_price == 100
    assert fetched["status"] == "closed"
    assert fetched["cost"] == 198.0
    assert closed == [fetched]
    assert canceled == fetched
    instance.trading.cancel_order_by_id.assert_called_once_with("order-1")
    request = instance.trading.get_orders.call_args.args[0]
    assert request.status == "closed"
    assert request.symbols == ["AAPL"]


@pytest.mark.parametrize(
    "symbol,client,prefix",
    [
        ("AAPL/USD", "stock_data", "stock"),
        ("BTC/USD", "crypto_data", "crypto"),
        ("AAPL260116C00200000/USD", "option_data", "option"),
    ],
)
def test_fetch_ticker_and_public_trades(symbol: str, client: str, prefix: str) -> None:
    instance = api()
    now = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    asset_id = instance.markets[symbol]["id"]
    quote = SimpleNamespace(
        timestamp=now, bid_price=99, bid_size=4, ask_price=101, ask_size=5
    )
    trade = SimpleNamespace(timestamp=now, price=100, size=2, id=12)
    data_client = getattr(instance, client)
    getattr(data_client, f"get_{prefix}_latest_quote").return_value = {asset_id: quote}
    getattr(data_client, f"get_{prefix}_latest_trade").return_value = {asset_id: trade}
    getattr(data_client, f"get_{prefix}_trades").return_value = SimpleNamespace(
        data={asset_id: [trade]}
    )

    ticker = instance.fetch_ticker(symbol)
    trades = instance.fetch_trades(symbol, limit=10)

    assert ticker["bid"] == 99.0
    assert ticker["ask"] == 101.0
    assert ticker["last"] == 100.0
    assert trades[0]["amount"] == 2.0
    assert trades[0]["cost"] == 200.0


def test_crypto_order_book_and_equity_l1_fallback() -> None:
    instance = api()
    now = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    book = SimpleNamespace(
        timestamp=now,
        bids=[SimpleNamespace(price=99, size=2)],
        asks=[SimpleNamespace(price=101, size=3)],
    )
    instance.crypto_data.get_crypto_latest_orderbook.return_value = {"BTC/USD": book}

    crypto_book = instance.fetch_l2_order_book("BTC/USD", 10)

    assert crypto_book["bids"] == [[99.0, 2.0]]
    assert crypto_book["asks"] == [[101.0, 3.0]]


@pytest.mark.asyncio
async def test_async_helpers_do_not_block_event_loop(monkeypatch) -> None:
    instance = _AlpacaAsyncApi.__new__(_AlpacaAsyncApi)
    expected = [[1, 2, 3, 4, 5, 6]]
    monkeypatch.setattr(_AlpacaApi, "fetch_ohlcv", lambda *args, **kwargs: expected)

    result = await instance.fetch_ohlcv("AAPL/USD", "1m")

    assert result == expected
