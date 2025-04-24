import csv
from datetime import datetime
from typing import Dict, List, Optional


class Transaction:
    def __init__(self, timestamp: datetime, changes: Dict[str, float], note: Optional[str] = None):
        """
        :param timestamp: datetime of the transaction (to the minute)
        :param changes: dict of currency symbol to amount changed (positive or negative)
        :param note: optional note for the transaction
        """
        self.timestamp = timestamp
        self.changes = changes
        self.note = note

class Wallet:
    def __init__(self, initial_holdings: Dict[str, float], start_time: datetime):
        """
        :param initial_holdings: dict of currency symbol to starting amount
        :param start_time: datetime of wallet creation
        """
        self.holdings = initial_holdings.copy()
        self.history = {start_time: self.holdings.copy()}
        self.transactions: List[Transaction] = []

    def transact(self, changes: Dict[str, float], timestamp: datetime, note: Optional[str] = None):
        """
        Apply a transaction to the wallet.
        :param changes: dict of currency symbol to amount to add/subtract
        :param timestamp: datetime of the transaction
        :param note: optional note for the transaction
        """
        for currency, amount in changes.items():
            self.holdings[currency] = self.holdings.get(currency, 0) + amount
        self.transactions.append(Transaction(timestamp, changes, note))
        self.history[timestamp] = self.holdings.copy()

    def get_value(self, prices: Dict[str, float]) -> float:
        """
        Get the total wallet value in USD given a dict of prices.
        :param prices: dict of currency symbol to USD price (e.g. {'ETH': 3200, 'BTC': 60000, 'USDC': 1})
        :return: total value in USD
        """
        return sum(self.holdings.get(cur, 0) * prices.get(cur, 0) for cur in self.holdings)

    def export_hourly_history_csv(self, prices_by_hour: Dict[datetime, Dict[str, float]], filename: str):
        """
        Export hourly closing wallet value to CSV.
        :param prices_by_hour: dict of datetime (hour) to dict of prices for that hour
        :param filename: output CSV file path
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'eth', 'btc', 'usdc', 'wallet_value_usd'])
            for hour, prices in sorted(prices_by_hour.items()):
                value = self.get_value(prices)
                writer.writerow([
                    hour.isoformat(),
                    self.holdings.get('ETH', 0),
                    self.holdings.get('BTC', 0),
                    self.holdings.get('USDC', 0),
                    value
                ])

    def export_transactions_csv(self, filename: str):
        """
        Export all transactions to CSV.
        :param filename: output CSV file path
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'currency', 'amount', 'note'])
            for tx in self.transactions:
                for currency, amount in tx.changes.items():
                    writer.writerow([
                        tx.timestamp.isoformat(),
                        currency,
                        amount,
                        tx.note or ''
                    ])