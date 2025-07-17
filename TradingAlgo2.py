import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ImprovedTradingAlgorithm:
    def __init__(self, symbols, initial_capital=100000, risk_per_trade=0.015, max_positions=3):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        self.dates = []
        self.market_regime = 'neutral'
        
    def fetch_data(self, period="2y"):
        """Fetch historical data for all symbols"""
        data = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if len(hist) > 100:  # Ensure we have enough data
                    data[symbol] = hist
                    print(f"Successfully fetched {len(hist)} days of data for {symbol}")
                else:
                    print(f"Insufficient data for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        return data
    
    def calculate_indicators(self, df):
        """Calculate enhanced technical indicators"""
        # Price-based indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD with improved parameters
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI with multiple timeframes
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        df['RSI_7'] = self.calculate_rsi(df['Close'], 7)
        
        # Bollinger Bands with dynamic period
        bb_period = 20
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility indicators
        df['ATR'] = self.calculate_atr(df, 14)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Volatility_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=50).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = self.calculate_obv(df)
        
        # Momentum indicators
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        df['Price_Change_20'] = df['Close'].pct_change(20)
        
        # Trend strength
        df['ADX'] = self.calculate_adx(df, 14)
        
        # Support/Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        # Fill NaN values with method='ffill' for better signal generation
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI with proper handling"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(df['Close'] * 0.02)  # Fill with 2% of price as default
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self.calculate_atr(df, 1)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
        
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period).mean()
        return adx.fillna(20)  # Fill with neutral value
    
    def detect_market_regime(self, df):
        """Detect market regime: bull, bear, or sideways"""
        try:
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            volatility_ratio = df['Volatility_Ratio'].iloc[-1]
            adx = df['ADX'].iloc[-1]
            
            # Handle NaN values
            if pd.isna(sma_50) or pd.isna(sma_200) or pd.isna(volatility_ratio) or pd.isna(adx):
                return 'neutral'
            
            # Trend strength
            if adx > 25:  # Strong trend
                if sma_50 > sma_200 and current_price > sma_50:
                    return 'bull'
                elif sma_50 < sma_200 and current_price < sma_50:
                    return 'bear'
            
            # Sideways/choppy market
            if volatility_ratio < 0.8 and adx < 20:
                return 'sideways'
            
            return 'neutral'
        except:
            return 'neutral'
    
    def enhanced_momentum_strategy(self, df):
        """Improved momentum strategy with relaxed thresholds"""
        signals = pd.Series(0, index=df.index)
        
        # Primary momentum signals with NaN handling
        macd_bullish = ((df['MACD'] > df['MACD_Signal']) & 
                       (df['MACD_Histogram'] > df['MACD_Histogram'].shift())).fillna(False)
        macd_bearish = ((df['MACD'] < df['MACD_Signal']) & 
                       (df['MACD_Histogram'] < df['MACD_Histogram'].shift())).fillna(False)
        
        # Moving average alignment (relaxed)
        ma_bullish = (df['SMA_10'] > df['SMA_20']).fillna(False)
        ma_bearish = (df['SMA_10'] < df['SMA_20']).fillna(False)
        
        # Momentum confirmation (relaxed thresholds)
        momentum_bullish = (df['Price_Change_5'] > 0.01).fillna(False)  # Reduced from 0.02
        momentum_bearish = (df['Price_Change_5'] < -0.01).fillna(False)  # Reduced from -0.02
        
        # Volume confirmation (relaxed)
        volume_confirm = (df['Volume_Ratio'] > 1.1).fillna(False)  # Reduced from 1.2
        
        # RSI filter (broader range)
        rsi_filter_buy = ((df['RSI_14'] > 30) & (df['RSI_14'] < 70)).fillna(False)
        rsi_filter_sell = ((df['RSI_14'] > 30) & (df['RSI_14'] < 70)).fillna(False)
        
        # Trend strength filter (relaxed)
        trend_strength = (df['ADX'] > 15).fillna(False)  # Reduced from 20
        
        # Buy signals (fewer confirmations needed)
        buy_condition = (
            macd_bullish & 
            ma_bullish & 
            momentum_bullish & 
            rsi_filter_buy
        )
        
        # Sell signals
        sell_condition = (
            macd_bearish & 
            ma_bearish & 
            momentum_bearish & 
            rsi_filter_sell
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def enhanced_mean_reversion_strategy(self, df):
        """Improved mean reversion strategy with relaxed thresholds"""
        signals = pd.Series(0, index=df.index)
        
        # Bollinger Band signals (relaxed)
        bb_oversold = (df['BB_Position'] < 0.2).fillna(False)  # Relaxed from 0.1
        bb_overbought = (df['BB_Position'] > 0.8).fillna(False)  # Relaxed from 0.9
        
        # RSI confirmation (relaxed)
        rsi_oversold = (df['RSI_14'] < 35).fillna(False)  # Relaxed from 25
        rsi_overbought = (df['RSI_14'] > 65).fillna(False)  # Relaxed from 75
        
        # Buy signals (oversold conditions)
        buy_condition = bb_oversold & rsi_oversold
        
        # Sell signals (overbought conditions)
        sell_condition = bb_overbought & rsi_overbought
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def calculate_position_size(self, symbol, entry_price, atr, signal_type):
        """Enhanced position sizing with Kelly criterion influence"""
        if len(self.positions) >= self.max_positions:
            return 0, 0
        
        # Base risk amount
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Handle NaN ATR
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02
        
        # Dynamic stop distance based on volatility
        stop_distance = max(atr * 1.5, entry_price * 0.02)  # Minimum 2% stop
        
        if signal_type == 1:  # Long position
            stop_loss = entry_price - stop_distance
        else:  # Short position
            stop_loss = entry_price + stop_distance
        
        # Position size based on risk
        position_size = risk_amount / stop_distance
        
        # Maximum position value (15% of capital - increased from 10%)
        max_position_value = self.current_capital * 0.15
        max_shares = max_position_value / entry_price
        position_size = min(position_size, max_shares)
        
        # Minimum position size check (reduced minimum)
        min_position_value = 500  # Reduced from $1000
        if position_size * entry_price < min_position_value:
            return 0, 0
        
        return position_size, stop_loss
    
    def should_exit_position(self, symbol, current_price, df):
        """Enhanced exit strategy with multiple conditions"""
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        
        # Time-based exit (hold for maximum 45 days - increased from 30)
        days_held = len(df) - df.index.get_loc(pos['entry_date'])
        if days_held > 45:
            return True
        
        # Profit target (1.5:1 risk-reward ratio - reduced from 2:1)
        entry_price = pos['entry_price']
        stop_distance = abs(entry_price - pos['stop_loss'])
        
        if pos['type'] == 'long':
            profit_target = entry_price + (stop_distance * 1.5)
            if current_price >= profit_target:
                return True
        else:
            profit_target = entry_price - (stop_distance * 1.5)
            if current_price <= profit_target:
                return True
        
        # Technical exit signals (relaxed)
        try:
            rsi = df['RSI_14'].iloc[-1]
            if not pd.isna(rsi):
                if pos['type'] == 'long' and rsi > 75:  # Reduced from 80
                    return True
                elif pos['type'] == 'short' and rsi < 25:  # Increased from 20
                    return True
        except:
            pass
        
        return False
    
    def execute_trade(self, symbol, price, signal, atr, date, df):
        """Execute trades with enhanced logic"""
        if signal == 0:
            return
        
        # Exit existing position if signal reverses
        if symbol in self.positions:
            current_pos = self.positions[symbol]
            if ((current_pos['type'] == 'long' and signal == -1) or 
                (current_pos['type'] == 'short' and signal == 1) or
                self.should_exit_position(symbol, price, df)):
                self.close_position(symbol, price, date)
        
        # Open new position
        if symbol not in self.positions:
            position_size, stop_loss = self.calculate_position_size(symbol, price, atr, signal)
            
            if position_size > 0:
                pos_type = 'long' if signal == 1 else 'short'
                position_value = position_size * price
                
                if position_value <= self.current_capital * 0.8:  # Reduced from 0.9 to ensure more capital available
                    self.positions[symbol] = {
                        'type': pos_type,
                        'shares': position_size,
                        'entry_price': price,
                        'stop_loss': stop_loss,
                        'entry_date': date,
                        'profit_target': price + (abs(price - stop_loss) * 1.5) if signal == 1 else price - (abs(price - stop_loss) * 1.5)
                    }
                    
                    self.current_capital -= position_value
                    
                    self.trades.append({
                        'symbol': symbol,
                        'action': 'open',
                        'type': pos_type,
                        'shares': position_size,
                        'price': price,
                        'date': date,
                        'value': position_value
                    })
    
    def close_position(self, symbol, price, date):
        """Close position with improved tracking"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        shares = pos['shares']
        entry_price = pos['entry_price']
        
        if pos['type'] == 'long':
            pnl = (price - entry_price) * shares
            self.current_capital += shares * price
        else:  # short
            pnl = (entry_price - price) * shares
            self.current_capital += shares * entry_price + pnl
        
        self.trades.append({
            'symbol': symbol,
            'action': 'close',
            'type': pos['type'],
            'shares': shares,
            'price': price,
            'date': date,
            'pnl': pnl,
            'entry_price': entry_price,
            'days_held': (date - pos['entry_date']).days if hasattr(date - pos['entry_date'], 'days') else 1
        })
        
        del self.positions[symbol]
    
    def backtest(self, data):
        """Enhanced backtesting with regime detection"""
        # Get common date range
        all_dates = None
        for symbol in data:
            if all_dates is None:
                all_dates = data[symbol].index
            else:
                all_dates = all_dates.intersection(data[symbol].index)
        
        all_dates = sorted(all_dates)
        
        for i, date in enumerate(all_dates):
            if i < 50:  # Reduced from 200 to allow more trading opportunities
                continue
                
            portfolio_value = self.current_capital
            
            # Calculate portfolio value and manage existing positions
            positions_to_close = []
            for symbol in list(self.positions.keys()):
                if date in data[symbol].index:
                    current_price = data[symbol].loc[date, 'Close']
                    pos = self.positions[symbol]
                    
                    if pos['type'] == 'long':
                        portfolio_value += pos['shares'] * current_price
                    else:
                        portfolio_value += pos['shares'] * pos['entry_price'] - (current_price - pos['entry_price']) * pos['shares']
                    
                    # Check exit conditions
                    if ((pos['type'] == 'long' and current_price <= pos['stop_loss']) or 
                        (pos['type'] == 'short' and current_price >= pos['stop_loss']) or
                        self.should_exit_position(symbol, current_price, data[symbol].loc[:date])):
                        positions_to_close.append((symbol, current_price, date))
            
            # Close flagged positions
            for symbol, price, close_date in positions_to_close:
                self.close_position(symbol, price, close_date)
            
            # Generate new signals
            for symbol in self.symbols:
                if date in data[symbol].index and symbol not in self.positions:
                    df = data[symbol].loc[:date]
                    if len(df) > 50:  # Reduced from 200
                        df = self.calculate_indicators(df)
                        
                        # Detect market regime
                        regime = self.detect_market_regime(df)
                        
                        # Get strategy signals
                        momentum_signal = self.enhanced_momentum_strategy(df).iloc[-1]
                        mean_reversion_signal = self.enhanced_mean_reversion_strategy(df).iloc[-1]
                        
                        # Combine strategies based on market regime
                        if regime == 'bull':
                            combined_signal = momentum_signal * 0.8 + mean_reversion_signal * 0.2
                        elif regime == 'bear':
                            combined_signal = momentum_signal * 0.7 + mean_reversion_signal * 0.3
                        elif regime == 'sideways':
                            combined_signal = momentum_signal * 0.3 + mean_reversion_signal * 0.7
                        else:  # neutral
                            combined_signal = momentum_signal * 0.5 + mean_reversion_signal * 0.5
                        
                        # Execute trade with relaxed threshold
                        if abs(combined_signal) > 0.3:  # Significantly reduced from 0.7
                            signal = 1 if combined_signal > 0 else -1
                            current_price = df['Close'].iloc[-1]
                            atr = df['ATR'].iloc[-1]
                            self.execute_trade(symbol, current_price, signal, atr, date, df)
            
            self.portfolio_value.append(portfolio_value)
            self.dates.append(date)
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.portfolio_value) < 2:
            return {}
        
        portfolio_returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
        # Basic metrics
        total_return = (self.portfolio_value[-1] / self.initial_capital - 1) * 100
        days_trading = len(self.portfolio_value)
        annualized_return = ((self.portfolio_value[-1] / self.initial_capital) ** (252 / days_trading) - 1) * 100
        
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return - 2) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak = pd.Series(self.portfolio_value).expanding().max()
        drawdown = (pd.Series(self.portfolio_value) - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Trade analysis
        closed_trades = [t for t in self.trades if t.get('pnl') is not None]
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Calmar Ratio': round(calmar_ratio, 2),
            'Maximum Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Profit Factor': round(profit_factor, 2),
            'Average Win ($)': round(avg_win, 2),
            'Average Loss ($)': round(avg_loss, 2),
            'Total Trades': len(closed_trades),
            'Final Portfolio Value': round(self.portfolio_value[-1], 2)
        }
    
    def plot_results(self):
        """Enhanced plotting with additional insights"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Portfolio value over time
        ax1.plot(self.dates, self.portfolio_value, label='Portfolio Value', linewidth=2, color='blue')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital', alpha=0.7)
        ax1.fill_between(self.dates, self.initial_capital, self.portfolio_value, 
                        where=[v >= self.initial_capital for v in self.portfolio_value], 
                        color='green', alpha=0.3, label='Profit')
        ax1.fill_between(self.dates, self.initial_capital, self.portfolio_value, 
                        where=[v < self.initial_capital for v in self.portfolio_value], 
                        color='red', alpha=0.3, label='Loss')
        ax1.set_title('Portfolio Performance Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        if len(self.portfolio_value) > 1:
            returns = pd.Series(self.portfolio_value).pct_change().dropna() * 100
            ax2.hist(returns, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax2.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
            ax2.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Daily Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Drawdown analysis
        peak = pd.Series(self.portfolio_value).expanding().max()
        drawdown = (pd.Series(self.portfolio_value) - peak) / peak * 100
        ax3.fill_between(self.dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax3.plot(self.dates, drawdown, color='darkred', linewidth=1)
        ax3.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Trade analysis
        closed_trades = [t for t in self.trades if t.get('pnl') is not None]
        if closed_trades:
            trade_pnls = [t['pnl'] for t in closed_trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
            ax4.bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('P&L ($)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_algorithm(self):
        """Main function to run the improved trading algorithm"""
        print("Starting Enhanced Trading Algorithm...")
        print("=" * 60)
        
        print("Fetching market data...")
        data = self.fetch_data()
        
        if not data:
            print("❌ No data available. Please check your symbols.")
            return
        
        print(f"✅ Successfully loaded data for {len(data)} symbols")
        print("🔄 Running enhanced backtest...")
        
        self.backtest(data)
        
        print("Calculating performance metrics...")
        metrics = self.calculate_performance_metrics()
        
        print("\n" + "=" * 60)
        print("ENHANCED TRADING ALGORITHM PERFORMANCE REPORT")
        print("=" * 60)
        
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        
        print(f"\nAlgorithm Statistics:")
        print(f"   • Active trading days: {len(self.dates)}")
        print(f"   • Current open positions: {len(self.positions)}")
        print(f"   • Maximum concurrent positions: {self.max_positions}")
        print(f"   • Risk per trade: {self.risk_per_trade * 100}%")
        
        print("\n Generating performance visualizations...")
        self.plot_results()
        
        return metrics

# Example usage with improved parameters
if __name__ == "__main__":
    # Diversified portfolio with strong stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'NVDA']
    
    # Initialize improved algorithm
    algorithm = ImprovedTradingAlgorithm(
        symbols=symbols,
        initial_capital=100000,
        risk_per_trade=0.02,  # Slightly increased for more trades
        max_positions=4  # Increased from 3
    )
    
    # Run the enhanced algorithm
    performance = algorithm.run_algorithm()