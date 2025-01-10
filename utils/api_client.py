import requests
import hashlib
import hmac
import base64
import time
from urllib.parse import urlencode
from collections import deque
import websocket
import json
import threading


class APIClient:
    def __init__(self, rest_api_url, api_key, api_secret):
        self.rest_api_url = rest_api_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.live_ticks = deque(maxlen=100)  # Buffer for live tick data

    def _sign_request(self, url_path, data):
        data['nonce'] = str(int(time.time() * 1000))
        post_data = urlencode(data)
        encoded = (data['nonce'] + post_data).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _make_request(self, endpoint, data=None, method="POST", is_private=True):
        url = self.rest_api_url + endpoint
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = data or {}

        if method == "GET" and data:
            url += f"?{urlencode(data)}"

        if is_private:
            api_sign = self._sign_request(endpoint, data)
            headers["API-Key"] = self.api_key
            headers["API-Sign"] = api_sign

        response = requests.request(method, url, headers=headers, data=urlencode(data) if method != "GET" else None, timeout=10)
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                return response.text
        else:
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

    def start_ticker_websocket(self, pair, callback):
        ws_url = "wss://ws.kraken.com"  # Correct WebSocket URL

        def on_message(ws, message):
            print(f"Message received: {message}")  # Debugging
            data = json.loads(message)
            if isinstance(data, list) and len(data) > 1:
                callback(data)

        def on_error(ws, error):
            print(f"WebSocket error: {error}")  # Debugging errors

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_status_code}, {close_msg}")  # Debugging closure

        def on_open(ws):
            subscription_message = {
                "event": "subscribe",
                "pair": [pair],  # Use correct Kraken pair format (e.g., "XBT/USD")
                "subscription": {"name": "ticker"},
            }
            ws.send(json.dumps(subscription_message))
            print(f"Subscribed to ticker feed for {pair}")

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.on_open = on_open
        threading.Thread(target=ws.run_forever, daemon=True).start()


    # --- Account and Funding Methods ---
    def get_account_balance(self):
        """
        Fetch the account balance.

        Returns:
        - dict: Account balances by currency.
        """
        return self._make_request("/0/private/Balance")

    def get_trade_balance(self, asset="ZUSD"):
        """
        Fetch the trade balance for a specific asset.

        Parameters:
        - asset (str): Asset (e.g., "ZUSD" for USD).

        Returns:
        - dict: Trade balance details.
        """
        return self._make_request("/0/private/TradeBalance", {"asset": asset})

    # --- Trading Methods ---
    def add_order(self, pair, action, ordertype, volume, price=None, price2=None):
        """
        Place an order.

        Parameters:
        - pair (str): Asset pair (e.g., "XBTUSD").
        - type (str): "buy" or "sell".
        - ordertype (str): Type of order (e.g., "limit", "market").
        - volume (float): Order volume.
        - price (float, optional): Price for limit orders.
        - price2 (float, optional): Secondary price for stop-limit orders.

        Returns:
        - dict: Order placement response.
        """
        data = {
            "pair": pair,
            "type": action,
            "ordertype": ordertype,
            "volume": volume,
            # "userref": userref,
        }
        if price:
            data["price"] = price
        if price2:
            data["price2"] = price2

        return self._make_request("/0/private/AddOrder", data)

    def cancel_order(self, pair, userref):
        """
        Cancel an existing order.

        Parameters:
        - txid (str): Transaction ID of the order.

        Returns:
        - dict: Cancellation response.
        """
        return self._make_request("/0/private/CancelOrder", {"pair": pair, "userref": userref})

    # --- Market Data Methods ---
    def get_ticker_info(self, pair: str):
        """
        Fetch ticker information for an asset pair.

        Parameters:
        - pair (str): Asset pair (e.g., "XBTUSD").

        Returns:
        - dict: Ticker information.
        """
        return self._make_request("/0/public/Ticker", {"pair": pair}, method="GET", is_private=False)

    def get_ohlc_data(self, pair, interval=1, since=None):
        """
        Fetch OHLC (Open/High/Low/Close) data for an asset pair.

        Parameters:
        - pair (str): Asset pair (e.g., "XBTUSD").
        - interval (int): Timeframe interval (in minutes).
        - since (int): Starting timestamp for historical in epoch format.

        Returns:
        - dict: OHLC data.
        """
        return self._make_request("/0/public/OHLC", {"pair": pair, "interval": interval, "since": since}, method="GET", is_private=False)
