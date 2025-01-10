import websocket
import json
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_ticker_message(message):
    """
    Callback function to process incoming ticker messages.
    """
    logger.info(f"Ticker Message: {message}")


def start_ticker_websocket(pair, callback):
    """
    Start a WebSocket connection to receive ticker data.
    """
    ws_url = "wss://ws.kraken.com"

    def on_message(ws, message):
        logger.info(f"Message received: {message}")
        data = json.loads(message)
        if isinstance(data, list) and len(data) > 1:
            callback(data)

    def on_error(ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        logger.warning(f"WebSocket closed with code {close_status_code}, message: {close_msg}")

    def on_open(ws):
        subscription_message = {
            "event": "subscribe",
            "pair": [pair],
            "subscription": {"name": "ticker"},
        }
        ws.send(json.dumps(subscription_message))
        logger.info(f"Subscribed to ticker feed for {pair}")

    # Initialize the WebSocketApp
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open

    # Run WebSocket in a separate thread
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()


# Main entry point
if __name__ == "__main__":
    # Start WebSocket for ticker feed
    start_ticker_websocket("XBT/USD", process_ticker_message)

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
