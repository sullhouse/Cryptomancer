from datetime import datetime, timedelta
from speculator import Speculator, RiskProfile
from exchange import Exchange

def get_fees():
    # Example: random flat and perc fees for each supported pair
    return {
        "ETH_USDC": {
            "flat_fee": 0,
            "perc_fee": .006
        },
        "BTC_USDC": {
            "flat_fee": .0000001,
            "perc_fee": .006
        },
        "USDC_ETH": {
            "flat_fee": 0,
            "perc_fee": .006
        },
        "USDC_BTC": {
            "flat_fee": .01,
            "perc_fee": .006
        }
    }

def main():
    # Initial holdings
    initial_holdings = {
        "ETH": 1.0,
        "BTC": 0,
        "USDC": 1000
    }
    start_time = datetime(2025, 4, 21, 2, 0, 0)
    end_time = datetime(2025, 4, 23, 5, 0, 0)

    risk_profile = RiskProfile(
        min_confidence=.1,
        min_time_since_last_swap=timedelta(1),
        max_daily_trades=24,
        trade_amount=.5,
        prediction_horizon_hours=1,
        always_max=True
    )

    exchange_fees = get_fees()
    slippage = .001

    spec = Speculator(
        initial_holdings=initial_holdings,
        start_time=start_time,
        exchange_fees=exchange_fees,
        slippage=slippage,
        risk_profile=risk_profile
    )

    # Cache predictions for the simulation period
    spec.cache_predictions(start_time, end_time)
    spec.exchange.cache_rates(start_time, end_time)

    print("Initial Holdings:", initial_holdings)
    print("Risk Profile:", risk_profile)
    print("Exchange Fees:", exchange_fees)
    print("Slippage:", slippage)

    current_time = start_time
    while current_time < end_time:
        result = spec.should_trade(current_time)
        print(f"{current_time}: {result}")
        print(f"Wallet Holdings: {spec.wallet.holdings}")
        print(f"Wallet Value: {spec.get_wallet_value_in_usdc(current_time)}")
        current_time += timedelta(hours=1)

    spec.wallet.export_hourly_history_csv({}, "wallet_history.csv")
    spec.wallet.export_transactions_csv("transactions.csv")

if __name__ == "__main__":
    from test import setup_environment
    setup_environment()
    main()