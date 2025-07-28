from unittest.mock import MagicMock

from tests.conftest import get_patched_exchange


def test_additional_exchange_init_alpaca(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.urls = {}
    default_conf["exchange"]["paper"] = True
    get_patched_exchange(mocker, default_conf, api_mock, exchange="alpaca")
    assert api_mock.urls["api"] == "https://paper-api.alpaca.markets"

    api_mock.urls = {}
    default_conf["exchange"]["paper"] = False
    get_patched_exchange(mocker, default_conf, api_mock, exchange="alpaca")
    assert api_mock.urls["api"] == "https://api.alpaca.markets"
