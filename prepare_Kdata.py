import pandas as pd
from datetime import datetime

print("📊 Preparing Kaggle Bitcoin Dataset")
print("=" * 50)

# Load YOUR dataset
print("\nLoading BitcoinTweets.csv...")
try:
    df = pd.read_csv('BitcoinTweets.csv')
    print(f"✓ Loaded {len(df)} tweets")
except FileNotFoundError:
    print("❌ Error: BitcoinTweets.csv not found!")
    print("Make sure the file is in your project folder")
    import sys
    sys.exit()

# Show what we have
print("\n📋 Columns in your dataset:")
print(df.columns.tolist())

print("\n🔍 First 3 rows:")
print(df.head(3))

print("\n📊 Dataset info:")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Standardize column names
column_mapping = {}

# Find text column
for col in df.columns:
    col_lower = col.lower()
    if 'text' in col_lower or 'tweet' in col_lower:
        column_mapping[col] = 'text'
        print(f"✓ Found text column: '{col}'")
        break

# Find date column
for col in df.columns:
    col_lower = col.lower()
    if 'date' in col_lower or 'created' in col_lower or 'timestamp' in col_lower:
        column_mapping[col] = 'date'
        print(f"✓ Found date column: '{col}'")
        break

# Rename
if column_mapping:
    df = df.rename(columns=column_mapping)

# Add missing columns
if 'tweet_id' not in df.columns:
    df['tweet_id'] = range(1, len(df) + 1)
if 'username' not in df.columns:
    df['username'] = 'user_' + df['tweet_id'].astype(str)
if 'likes' not in df.columns:
    df['likes'] = 0
if 'retweets' not in df.columns:
    df['retweets'] = 0
if 'replies' not in df.columns:
    df['replies'] = 0

# Parse dates
print("\n📅 Processing dates...")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Get last 12 months of data
    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=365)
    
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    print(f"✓ Filtered to last 365 days: {len(df)} tweets")
else:
    print("⚠️ Creating synthetic dates...")
    df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5min')

# Remove duplicates
df = df.drop_duplicates(subset=['text'])
print(f"✓ Removed duplicates, {len(df)} tweets remaining")

# Keep only what we need
required_cols = ['date', 'tweet_id', 'text', 'username', 'likes', 'retweets', 'replies']
df = df[required_cols]

# Remove empty tweets
df = df[df['text'].notna()]
df = df[df['text'].str.len() > 5]

print(f"✓ Final dataset: {len(df)} tweets")

# Save
df.to_csv('bitcoin_tweets.csv', index=False)

print("\n" + "=" * 50)
print("✅ SUCCESS!")
print("=" * 50)
print(f"📁 Saved to: bitcoin_tweets.csv")
print(f"Total tweets: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

print("\n📊 Sample of cleaned data:")
print(df[['date', 'text']].head(5))

print("\n✅ Next step: Run 'python collect_prices.py'")