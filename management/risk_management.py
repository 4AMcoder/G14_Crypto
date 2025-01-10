class RiskManager:
    def __init__(self, max_drawdown=0.2, max_risk_per_trade=0.01):
        """
        Initialize the RiskManager.

        Parameters:
        - max_drawdown (float): Maximum allowable portfolio drawdown (e.g., 0.2 for 20%).
        - max_risk_per_trade (float): Maximum risk per trade as a percentage of total capital.
        """
        self.max_drawdown = max_drawdown
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_position_size(self, capital, entry_price, stop_loss_price):
        """
        Calculate the allowable position size based on risk constraints.

        Parameters:
        - capital (float): Total available capital.
        - entry_price (float): Entry price of the trade.
        - stop_loss_price (float): Stop-loss price for the trade.

        Returns:
        - float: Position size in units of the asset.
        """
        risk_per_unit = abs(entry_price - stop_loss_price)
        risk_per_trade = capital * self.max_risk_per_trade
        position_size = risk_per_trade / risk_per_unit
        return position_size

    def is_within_drawdown_limit(self, current_value, peak_value):
        """
        Check if the portfolio is within the allowable drawdown.

        Parameters:
        - current_value (float): Current portfolio value.
        - peak_value (float): Peak portfolio value.

        Returns:
        - bool: True if within drawdown limit, False otherwise.
        """
        drawdown = (peak_value - current_value) / peak_value
        return drawdown <= self.max_drawdown
