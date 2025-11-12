import pandas as pd

print("🔧 Fixing Data Merge")
print("=" * 50)

# Load both files
tweets = pd.read_csv('tweets_cleaned.csv')
daily = pd.read_csv('daily_sentiment.csv')
prices = pd.read_csv('bitcoin_prices.csv')

print("\n📅 Date Ranges:")
tweets['date'] = pd.to_datetime(tweets['date'])
print(f"Tweets: {tweets['date'].min().date()} to {tweets['date'].max().date()}")

daily['date'] = pd.to_datetime(daily['date'])
print(f"Daily sentiment: {daily['date'].min()} to {daily['date'].max()}")

prices['date'] = pd.to_datetime(prices['date'])
print(f"Prices: {prices['date'].min().date()} to {prices['date'].max().date()}")

# They don't overlap! Let's download prices for the tweet date range
print("\n💡 Solution: Downloading prices for tweet date range...")

import yfinance as yf

tweet_start = tweets['date'].min().date()
tweet_end = tweets['date'].max().date()

print(f"Downloading BTC prices from {tweet_start} to {tweet_end}...")

btc = yf.download('BTC-USD', start=tweet_start, end=tweet_end, progress=False)

if len(btc) > 0:
    print(f"✓ Downloaded {len(btc)} days")
    
    # Process
    btc = btc.reset_index()
    
    # Flatten columns
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = ['_'.join(map(str, col)).strip('_') if isinstance(col, tuple) else col for col in btc.columns]
    
    # Rename
    new_cols = []
    for col in btc.columns:
        col_str = str(col).lower()
        if 'date' in col_str:
            new_cols.append('date')
        elif 'open' in col_str:
            new_cols.append('open')
        elif 'high' in col_str:
            new_cols.append('high')
        elif 'low' in col_str:
            new_cols.append('low')
        elif 'close' in col_str and 'adj' not in col_str:
            new_cols.append('close')
        elif 'volume' in col_str:
            new_cols.append('volume')
        else:
            new_cols.append(col)
    
    btc.columns = new_cols
    
    # Keep needed columns
    cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in btc.columns]
    btc = btc[cols]
    
    # Add features
    btc['price_change'] = btc['close'] - btc['open']
    btc['price_change_pct'] = (btc['price_change'] / btc['open']) * 100
    btc['label'] = (btc['price_change'] > 0).astype(int)
    btc['date'] = pd.to_datetime(btc['date']).dt.date
    
    # Save updated prices
    btc.to_csv('bitcoin_prices.csv', index=False)
    print(f"✓ Saved updated bitcoin_prices.csv")
    
    # Now merge
    print("\n🔗 Merging data...")
    daily['date'] = daily['date'].dt.date
    
    final_data = pd.merge(btc, daily, on='date', how='left')
    final_data = final_data.fillna(0)
    
    final_data.to_csv('final_dataset.csv', index=False)
    
    print(f"✓ Saved final_dataset.csv")
    print(f"  Total rows: {len(final_data)}")
    print(f"  Rows with tweets: {(final_data['tweet_count'] > 0).sum()}")
    
    print("\n📊 Sample:")
    print(final_data[['date', 'close', 'avg_sentiment', 'tweet_count', 'label']].head(10))
    
else:
    print("❌ No price data available for this date range")
    print("Your tweets are from a time period where BTC price data may not be available")