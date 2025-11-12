import pandas as pd

print("🔍 Checking Data Files")
print("=" * 50)

# Check final_dataset.csv
print("\n📄 final_dataset.csv:")
df = pd.read_csv('final_dataset.csv')
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nTweet count stats:")
print(f"  Rows with tweets > 0: {(df['tweet_count'] > 0).sum()}")
print(f"  Rows with tweets = 0: {(df['tweet_count'] == 0).sum()}")
print(f"  Min: {df['tweet_count'].min()}")
print(f"  Max: {df['tweet_count'].max()}")

print("\n" + "=" * 50)