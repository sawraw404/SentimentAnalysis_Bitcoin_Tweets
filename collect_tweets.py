import pandas as pd
import random
from datetime import datetime, timedelta

print("🐦 Bitcoin Tweet Collector")
print("=" * 50)
print("Generating sample Bitcoin tweets for your project...\n")

# Tweet templates for realistic data
positive_phrases = [
    "Bitcoin breaking resistance levels! 🚀",
    "BTC looking bullish today",
    "Great momentum in the crypto market",
    "Bitcoin to the moon 🌙",
    "Buying more BTC, feeling confident",
    "Bitcoin rally continues strong",
    "Best time to invest in Bitcoin",
    "BTC hitting new highs soon",
    "Crypto bulls taking over",
    "Bitcoin adoption growing fast"
]

negative_phrases = [
    "Bitcoin crash incoming?",
    "BTC dropping fast, sold my position",
    "Crypto market bleeding red today",
    "Bitcoin losing support levels",
    "Bearish on BTC right now",
    "Bitcoin price falling hard",
    "Market panic, everyone selling",
    "BTC could go lower from here",
    "Crypto winter is back",
    "Bitcoin correction overdue"
]

neutral_phrases = [
    "Bitcoin price stable at current levels",
    "BTC trading sideways today",
    "Monitoring Bitcoin closely",
    "Bitcoin market update for today",
    "BTC price analysis thread",
    "Crypto market consolidating",
    "Bitcoin holding support",
    "Interesting Bitcoin charts today",
    "BTC volume looking normal",
    "Bitcoin in accumulation phase"
]

# Generate sample tweets
sample_tweets = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 10, 31)
current_date = start_date
tweet_id = 1

print("Generating tweets...")

while current_date <= end_date:
    # 35-55 tweets per day
    tweets_per_day = random.randint(35, 55)
    
    for _ in range(tweets_per_day):
        # Random sentiment distribution (roughly realistic)
        rand = random.random()
        if rand < 0.35:  # 35% positive
            text = random.choice(positive_phrases)
            sentiment_type = 'positive'
        elif rand < 0.65:  # 30% negative
            text = random.choice(negative_phrases)
            sentiment_type = 'negative'
        else:  # 35% neutral
            text = random.choice(neutral_phrases)
            sentiment_type = 'neutral'
        
        # Add random hashtags
        hashtags = random.choice([
            '#Bitcoin', '#BTC', '#Crypto', '#Cryptocurrency',
            '#BitcoinNews', '#CryptoNews', '#HODL'
        ])
        
        # Random time during the day
        tweet_time = current_date + timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Engagement metrics (roughly correlated with sentiment)
        if sentiment_type == 'positive':
            likes = random.randint(50, 1500)
            retweets = random.randint(10, 500)
        elif sentiment_type == 'negative':
            likes = random.randint(30, 1000)
            retweets = random.randint(5, 300)
        else:
            likes = random.randint(10, 500)
            retweets = random.randint(2, 150)
        
        sample_tweets.append({
            'date': tweet_time,
            'tweet_id': tweet_id,
            'text': f"{text} {hashtags}",
            'username': f"crypto_user_{random.randint(1000, 9999)}",
            'likes': likes,
            'retweets': retweets,
            'replies': random.randint(0, 200)
        })
        
        tweet_id += 1
    
    current_date += timedelta(days=1)
    
    # Progress update
    if current_date.day == 1:
        print(f"  ✓ {current_date.strftime('%B %Y')}")

# Create DataFrame
df_tweets = pd.DataFrame(sample_tweets)

# Save to CSV
df_tweets.to_csv('bitcoin_tweets.csv', index=False)

print("\n" + "=" * 50)
print(f"✅ SUCCESS! Generated {len(df_tweets)} tweets")
print(f"📁 Saved to: bitcoin_tweets.csv")
print(f"\nDate range: {df_tweets['date'].min()} to {df_tweets['date'].max()}")
print(f"Total days: {(end_date - start_date).days + 1}")
print(f"Average tweets per day: {len(df_tweets) / ((end_date - start_date).days + 1):.1f}")

print("\n📊 Sample tweets:")
print(df_tweets[['date', 'text', 'likes', 'retweets']].head(5))

print("\n" + "=" * 50)
print("✅ Ready for next step!")
print("Run: python collect_prices.py")