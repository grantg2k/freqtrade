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
