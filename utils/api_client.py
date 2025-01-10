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
from typing import Callable, Dict, Optional, Union


class APIClient:
    def __init__(self, rest_api_url: str, api_key: str, api_secret: str) -> None:
        self.rest_api_url: str = rest_api_url
        self.api_key: str = api_key
        self.api_secret: str = api_secret
        self.live_ticks: deque = deque(maxlen=100)  # Buffer for live tick data

    def _sign_request(self, url_path: str, data: Dict[str, Union[str, int]]) -> str:
        data['nonce'] = str(int(time.time() * 1000))
        post_data = urlencode(data)
        encoded = (data['nonce'] + post_data).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _make_request(
        self,
        endpoint: str,
        data: Optional[Dict[str, Union[str, int, float]]] = None,
        method: str = "POST",
        is_private: bool = True
    ) -> Union[Dict, str]:
        url: str = self.rest_api_url + endpoint
        headers: Dict[str, str] = {"Content-Type": "application/x-www-form-urlencoded"}
        data = data or {}

        if method == "GET" and data:
            url += f"?{urlencode(data)}"

        if is_private:
            api_sign: str = self._sign_request(endpoint, data)
            headers["API-Key"] = self.api_key
            headers["API-Sign"] = api_sign

        response = requests.request(
            method,
            url,
            headers=headers,
            data=urlencode(data) if method != "GET" else None,
            timeout=10,
        )
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                return response.text
        else:
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

    def start_ticker_websocket(self, pair: str, callback: Callable[[Dict], None]) -> None:
        ws_url: str = "wss://ws.kraken.com"

        def on_message(ws: websocket.WebSocketApp, message: str) -> None:
            print(f"Message received: {message}")
            data = json.loads(message)
            if isinstance(data, list) and len(data) > 1:
                callback(data)

        def on_error(ws: websocket.WebSocketApp, error: str) -> None:
            print(f"WebSocket error: {error}")

        def on_close(ws: websocket.WebSocketApp, close_status_code: Optional[int], close_msg: Optional[str]) -> None:
            print(f"WebSocket closed: {close_status_code}, {close_msg}")

        def on_open(ws: websocket.WebSocketApp) -> None:
            subscription_message = {
                "event": "subscribe",
                "pair": [pair],
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
    def get_account_balance(self) -> Dict:
        return self._make_request("/0/private/Balance")

    def get_trade_balance(self, asset: str = "ZUSD") -> Dict:
        return self._make_request("/0/private/TradeBalance", {"asset": asset})

    # --- Trading Methods ---
    def add_order(
        self,
        pair: str,
        action: str,
        ordertype: str,
        volume: float,
        price: Optional[float] = None,
        price2: Optional[float] = None,
    ) -> Dict:
        data = {
            "pair": pair,
            "type": action,
            "ordertype": ordertype,
            "volume": volume,
        }
        if price:
            data["price"] = price
        if price2:
            data["price2"] = price2

        return self._make_request("/0/private/AddOrder", data)

    def cancel_order(self, pair: str, userref: str) -> Dict:
        return self._make_request("/0/private/CancelOrder", {"pair": pair, "userref": userref})

    # --- Market Data Methods ---
    def get_ticker_info(self, pair: str) -> Dict:
        return self._make_request("/0/public/Ticker", {"pair": pair}, method="GET", is_private=False)

    def get_ohlc_data(
        self, pair: str, interval: int = 1, since: Optional[int] = None
    ) -> Dict:
        return self._make_request(
            "/0/public/OHLC", {"pair": pair, "interval": interval, "since": since}, method="GET", is_private=False
        )
