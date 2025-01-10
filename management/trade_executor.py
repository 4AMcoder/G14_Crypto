import requests

class TradeExecutor:
    def __init__(self, api_client):
        """
        Initialize the TradeExecutor.

        Parameters:
        - api_client (APIClient): An instance of the APIClient to interact with the exchange.
        """
        self.api_client = api_client

    def place_market_order(self, side, volume, pair="BTCUSD"):
        """
        Place a market order.

        Parameters:
        - side (str): "buy" or "sell".
        - volume (float): Amount to trade.
        - pair (str): Trading pair.

        Returns:
        - dict: Response from the API.
        """
        return self.api_client.place_order(side=side, volume=volume, order_type="market", pair=pair)

    def place_limit_order(self, side, volume, price, pair="BTCUSD"):
        """
        Place a limit order.

        Parameters:
        - side (str): "buy" or "sell".
        - volume (float): Amount to trade.
        - price (float): Price at which to place the order.
        - pair (str): Trading pair.

        Returns:
        - dict: Response from the API.
        """
        return self.api_client.place_order(side=side, volume=volume, price=price, order_type="limit", pair=pair)

    def cancel_order(self, order_id):
        """
        Cancel an existing order.

        Parameters:
        - order_id (str): ID of the order to cancel.

        Returns:
        - dict: Response from the API.
        """
        url = f"{self.api_client.rest_api_url}/private/CancelOrder"
        payload = {"txid": order_id}
        headers = {"API-Key": self.api_client.api_key}
        response = requests.post(url, headers=headers, data=payload)
        return response.json()
