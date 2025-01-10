import pandas as pd

class Portfolio:
    def __init__(self, api_client):
        """
        Initialize the Portfolio.

        Parameters:
        - api_client (APIClient): API client to fetch the initial cash balance.
        """
        self.api_client = api_client
        self.cash = self.get_initial_cash_balance()
        self.positions = {}  # Dictionary to track open positions {pair: {volume, avg_price}}
        self.history = []    # List to track all transactions

    def get_initial_cash_balance(self):
        """
        Fetch the initial cash balance from the API.

        Returns:
        - float: Current cash balance.
        """
        account_balance = self.api_client.get_account_balance()  # Fetch account balance from the API
        cash_balance = float(account_balance.get("GBP", 0))  # Assume USD is the base currency
        return cash_balance

    def update_cash(self, amount):
        """
        Update the cash balance.

        Parameters:
        - amount (float): Positive for cash inflows, negative for outflows.
        """
        self.cash += amount

    def add_position(self, pair, volume, price):
        """
        Add a position or update an existing one.

        Parameters:
        - pair (str): Trading pair (e.g., "BTCUSD").
        - volume (float): Volume of the trade.
        - price (float): Trade price.
        """
        if pair in self.positions:
            # Update average price and volume for the position
            existing_volume = self.positions[pair]['volume']
            avg_price = self.positions[pair]['avg_price']
            new_avg_price = ((avg_price * existing_volume) + (price * volume)) / (existing_volume + volume)
            self.positions[pair]['volume'] += volume
            self.positions[pair]['avg_price'] = new_avg_price
        else:
            # Add a new position
            self.positions[pair] = {'volume': volume, 'avg_price': price}

    def remove_position(self, pair, volume, price):
        """
        Reduce or close a position.

        Parameters:
        - pair (str): Trading pair (e.g., "BTCUSD").
        - volume (float): Volume to close.
        - price (float): Trade price.
        """
        if pair in self.positions:
            existing_volume = self.positions[pair]['volume']
            if volume > existing_volume:
                raise ValueError("Cannot close more volume than exists in position.")

            # Update or remove position
            self.positions[pair]['volume'] -= volume
            if self.positions[pair]['volume'] == 0:
                del self.positions[pair]

            # Update cash balance
            self.update_cash(volume * price)

    def record_transaction(self, side, pair, volume, price):
        """
        Record a transaction in the portfolio history.

        Parameters:
        - side (str): "buy" or "sell".
        - pair (str): Trading pair.
        - volume (float): Volume of the trade.
        - price (float): Trade price.
        """
        transaction = {
            'side': side,
            'pair': pair,
            'volume': volume,
            'price': price,
            'timestamp': pd.Timestamp.now()
        }
        self.history.append(transaction)

    def calculate_portfolio_value(self, market_prices):
        """
        Calculate the total portfolio value.

        Parameters:
        - market_prices (dict): Current market prices {pair: price}.

        Returns:
        - float: Total portfolio value.
        """
        total_value = self.cash
        for pair, position in self.positions.items():
            if pair in market_prices:
                total_value += position['volume'] * market_prices[pair]
        return total_value

    def get_portfolio_summary(self):
        """
        Get a summary of the current portfolio.

        Returns:
        - dict: Summary of cash, positions, and portfolio value.
        """
        return {
            'cash': self.cash,
            'positions': self.positions,
            'history': self.history
        }
