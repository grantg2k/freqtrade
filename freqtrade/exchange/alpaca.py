"""Alpaca exchange subclass."""

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Alpaca(Exchange):
    """Alpaca exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,
    }

    def additional_exchange_init(self) -> None:
        """Set API domain based on paper trading configuration."""
        # ``paper`` defaults to True to use the paper trading API
        paper = self._config.get("paper", True)
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        try:
            if hasattr(self._api, "urls"):
                self._api.urls.setdefault("api", base_url)
                self._api.urls["api"] = base_url
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to set Alpaca base url: %s", e)
