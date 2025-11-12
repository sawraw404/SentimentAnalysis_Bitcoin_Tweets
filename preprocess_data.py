import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("📥 Downloading language data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("\n🧹 Starting Data Preprocessing...")
print("=" * 50)

print("\n[1/5] Loading tweets...")
tweets = pd.read_csv('BitcoinTweets.csv', nrows=50000)
print(f"✓ Loaded {len(tweets)} tweets")
print(f"✓ Columns: {tweets.columns.tolist()}")

# --- Clean Text ---
print("\n[2/5] Cleaning tweet text...")

def clean_tweet(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([w for w in words if w not in stop_words and len(w) > 2])
    
    return text

batch_size = 5000
cleaned_texts = []

for i in range(0, len(tweets), batch_size):
    batch = tweets.iloc[i:i+batch_size]['text']
    cleaned_batch = batch.apply(clean_tweet)
    cleaned_texts.extend(cleaned_batch.tolist())
    
    if (i + batch_size) % 10000 == 0:
        print(f"  ✓ Cleaned {min(i+batch_size, len(tweets))}/{len(tweets)} tweets")

tweets['cleaned_text'] = cleaned_texts
tweets = tweets[tweets['cleaned_text'].str.len() > 10]
print(f"✓ Kept {len(tweets)} tweets")

# --- Sentiment ---
print("\n[3/5] Calculating sentiment...")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    if not text:
        return 0
    return analyzer.polarity_scores(text)['compound']

def get_sentiment_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

sentiment_scores = []
for i in range(0, len(tweets), batch_size):
    batch = tweets.iloc[i:i+batch_size]['cleaned_text']
    batch_scores = batch.apply(get_sentiment_score)
    sentiment_scores.extend(batch_scores.tolist())
    
    if (i + batch_size) % 10000 == 0:
        print(f"  ✓ Processed {min(i+batch_size, len(tweets))}/{len(tweets)}")

tweets['sentiment_score'] = sentiment_scores
tweets['sentiment_label'] = tweets['sentiment_score'].apply(get_sentiment_label)

print("\n📊 Sentiment Distribution:")
print(tweets['sentiment_label'].value_counts())

# --- Aggregate ---
print("\n[4/5] Aggregating by day...")

tweets['date'] = pd.to_datetime(tweets['date']).dt.date

# Build aggregation based on available columns
agg_dict = {
    'sentiment_score': 'mean',
    'text': 'count'
}

# Add engagement metrics if they exist
for col in ['user_followers', 'user_friends', 'user_favourites']:
    if col in tweets.columns:
        agg_dict[col] = 'mean'

daily_sentiment = tweets.groupby('date').agg(agg_dict).reset_index()

# Rename columns
col_names = ['date', 'avg_sentiment', 'tweet_count']
for col in ['user_followers', 'user_friends', 'user_favourites']:
    if col in agg_dict:
        col_names.append(f'avg_{col.replace("user_", "")}')

daily_sentiment.columns = col_names

# Add dummy engagement columns if missing
if 'avg_likes' not in daily_sentiment.columns:
    daily_sentiment['avg_likes'] = 0
if 'avg_retweets' not in daily_sentiment.columns:
    daily_sentiment['avg_retweets'] = 0

print(f"✓ Created {len(daily_sentiment)} days")

# --- Merge with Prices ---
print("\n[5/5] Merging with prices...")

prices = pd.read_csv('bitcoin_prices.csv')
prices['date'] = pd.to_datetime(prices['date']).dt.date

final_data = pd.merge(prices, daily_sentiment, on='date', how='left')
final_data = final_data.fillna(0)

# --- Save ---
print("\n💾 Saving...")

tweets.to_csv('tweets_cleaned.csv', index=False)
print("✓ tweets_cleaned.csv")

daily_sentiment.to_csv('daily_sentiment.csv', index=False)
print("✓ daily_sentiment.csv")

final_data.to_csv('final_dataset.csv', index=False)
print("✓ final_dataset.csv")

# --- Summary ---
print("\n" + "=" * 50)
print("✅ COMPLETE!")
print("=" * 50)
print(f"Tweets: {len(tweets)}")
print(f"Days: {len(final_data)}")

pos = (tweets['sentiment_label']=='positive').sum()
neg = (tweets['sentiment_label']=='negative').sum()
neu = (tweets['sentiment_label']=='neutral').sum()

print(f"\n😊 Sentiment:")
print(f"  Positive: {pos} ({pos/len(tweets)*100:.1f}%)")
print(f"  Negative: {neg} ({neg/len(tweets)*100:.1f}%)")
print(f"  Neutral: {neu} ({neu/len(tweets)*100:.1f}%)")

up = (final_data['label']==1).sum()
print(f"\n📈 Price UP: {up} days ({up/len(final_data)*100:.1f}%)")

print("\n✅ Next: python model1_logistic_regression.py")