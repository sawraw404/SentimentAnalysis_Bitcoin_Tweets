import yfinance as yf
import pandas as pd
import os

print("💰 Bitcoin Price Data Collection")
print("=" * 50)

# Check which CSV file exists
if os.path.exists('bitcoin_tweets.csv'):
    tweet_file = 'bitcoin_tweets.csv'
elif os.path.exists('BitcoinTweets.csv'):
    tweet_file = 'BitcoinTweets.csv'
else:
    print("❌ No tweet file found!")
    tweet_start = '2016-01-01'
    tweet_end = '2019-03-31'

# Read tweets
if 'tweet_file' in locals():
    print(f"\n📥 Loading {tweet_file}...")
    try:
        chunks = []
        for chunk in pd.read_csv(tweet_file, chunksize=10000):
            chunks.append(chunk)
        tweets = pd.concat(chunks, ignore_index=True)
        print(f"✓ Loaded {len(tweets)} tweets")
        
        tweets['date'] = pd.to_datetime(tweets['date'], errors='coerce')
        tweets = tweets.dropna(subset=['date'])
        tweet_start = tweets['date'].min().date()
        tweet_end = tweets['date'].max().date()
    except:
        print("⚠️ Using default dates")
        tweet_start = '2016-01-01'
        tweet_end = '2019-03-31'

print(f"\n📅 Date range: {tweet_start} to {tweet_end}")

# Download Bitcoin prices
print(f"\n💰 Downloading BTC prices...")

btc = yf.download('BTC-USD', start=tweet_start, end=tweet_end, progress=False)

if len(btc) == 0:
    btc = yf.download('BTC-USD', start='2016-01-01', end='2019-03-31', progress=False)

print(f"✓ Downloaded {len(btc)} days")

# Reset index
btc = btc.reset_index()

# DEBUG: Show actual columns
print(f"\n🔍 Actual columns in data:")
for i, col in enumerate(btc.columns):
    print(f"   {i}: {col} (type: {type(col)})")

# Flatten multi-index columns if needed
if isinstance(btc.columns, pd.MultiIndex):
    print("\n⚙️ Flattening multi-index columns...")
    btc.columns = ['_'.join(map(str, col)).strip('_') if isinstance(col, tuple) else col for col in btc.columns]

# Now rename with simpler approach
print("\n⚙️ Renaming columns...")
new_columns = []
for col in btc.columns:
    col_str = str(col).lower()
    if 'date' in col_str:
        new_columns.append('date')
    elif 'open' in col_str:
        new_columns.append('open')
    elif 'high' in col_str:
        new_columns.append('high')
    elif 'low' in col_str:
        new_columns.append('low')
    elif 'close' in col_str and 'adj' not in col_str:
        new_columns.append('close')
    elif 'adj' in col_str:
        new_columns.append('adj_close')
    elif 'volume' in col_str:
        new_columns.append('volume')
    else:
        new_columns.append(col)

btc.columns = new_columns

print(f"✓ New columns: {btc.columns.tolist()}")

# Select only columns we have
available_cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in btc.columns]
btc = btc[available_cols]

# Add features
btc['price_change'] = btc['close'] - btc['open']
btc['price_change_pct'] = (btc['price_change'] / btc['open']) * 100
btc['label'] = (btc['price_change'] > 0).astype(int)
btc['date'] = pd.to_datetime(btc['date']).dt.date

# Save
btc.to_csv('bitcoin_prices.csv', index=False)

print("\n" + "=" * 50)
print("✅ SUCCESS!")
print("=" * 50)
print(f"📁 Saved: bitcoin_prices.csv")
print(f"Total days: {len(btc)}")
print(f"Range: {btc['date'].min()} to {btc['date'].max()}")

print("\n📊 Sample:")
print(btc.head())

up = (btc['label']==1).sum()
print(f"\n📈 Days UP: {up} ({up/len(btc)*100:.1f}%)")

print("\n✅ Next: python preprocess_data.py")