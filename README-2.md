# 📈 Portfolio Trading Algorithm
# First time building a trading algorithm


This project implements a backtestable portfolio trading algorithm using Python and financial data from Yahoo Finance. The algorithm incorporates common technical indicators like Exponential Moving Averages (EMAs) and Relative Strength Index (RSI), with built-in risk management techniques such as volatility targeting and trailing stop-loss.

## 🧠 Features

- EMA crossover strategy
- RSI filter for entry signals
- Volatility-adjusted position sizing
- Trailing stop-loss mechanism
- Performance evaluation (returns, Sharpe ratio, drawdown)
- Visual backtest plots

## 🔧 Requirements

Install required packages:

```bash
pip install yfinance numpy pandas matplotlib
```

## 🚀 Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/portfolio-trading-algorithm.git
cd portfolio-trading-algorithm
```

2. Run the algorithm:

```bash
python TradingAlgo2.py
```

3. You can change the ticker symbol and strategy parameters inside the script:
   ```python
   ticker = 'TSLA'
   short_window = 50
   long_window = 200
   rsi_period = 14
   stop_pct = 0.20
   target_vol = 0.10
   vol_window = 20
   ```

## 📊 Output

- Buy/sell signal markers on price chart
- Equity curve of strategy vs buy-and-hold
- Printed metrics including:
  - Total Return
  - Annualized Return
  - Sharpe Ratio
  - Max Drawdown

## 📁 File Structure

```
.
├── TradingAlgo2.py        # Main trading algorithm
├── README.md              # Project documentation
├── .gitignore             # (Optional) Git ignore rules
```

## 📌 Notes

- Historical data is fetched using `yfinance`.
- This strategy is for educational purposes and not financial advice.
- You can extend the script to handle multiple tickers or apply portfolio optimization.

## 📃 License

MIT License © [Your Name]
