import logging
from utils.api_client import APIClient
from utils.load_keys import load_secrets

secrets = load_secrets()

API_URL = "https://api.kraken.com/"

def actual_balance():
    """
    Test retrieving the actual balance via the REST API.
    Requires valid API keys and live connection.
    """
    logger = logging.getLogger("test_logger")
    logging.basicConfig(level=logging.INFO)

    client = APIClient(
        rest_api_url=API_URL,
        api_key=secrets["rest_api_key"],
        api_secret=secrets["rest_api_secret"],
    )

    try:
        balance = client.get_account_balance()
        print("Account Balance:", balance)
    except Exception as e:
        print("Error fetching account balance:", e)


def get_ticker_info():
    logger = logging.getLogger("test_logger")
    logging.basicConfig(level=logging.INFO)
    client = APIClient(
        rest_api_url=API_URL,
        api_key=secrets["rest_api_key"],
        api_secret=secrets["rest_api_secret"],)
    
    ticker_info = client.get_ticker_info("BTCUSD")
    print("info:", ticker_info)


def get_ticks():
    logger = logging.getLogger("test_logger")
    logging.basicConfig(level=logging.INFO)
    client = APIClient(
        rest_api_url=API_URL,
        api_key=secrets["rest_api_key"],
        api_secret=secrets["rest_api_secret"],)
    
    ticks = client.get_ohlc_data("BTCUSD", 240, 1732683600)
    print("info:", ticks)


def buysummat():
    logger = logging.getLogger("test_logger")
    logging.basicConfig(level=logging.INFO)
    client = APIClient(
        rest_api_url=API_URL,
        api_key=secrets["rest_api_key"],
        api_secret=secrets["rest_api_secret"],)
    
    ticks = client.add_order("XBTUSD", "buy", "limit", 0.01, price=100000)
    print("info:", ticks)

def process_ticker_message(message):
    if isinstance(message, list) and len(message) > 1:
        ticker_data = message[1]
        if "c" in ticker_data:
            ask_price = ticker_data["c"][0]
            print(f"XBT/USD Ask Price: {ask_price}")


if __name__ == "__main__":
    client = APIClient("https://api.kraken.com/0", secrets["rest_api_key"], secrets["rest_api_secret"])
    client.start_ticker_websocket("XBT/USD", process_ticker_message)

    get_ticks()