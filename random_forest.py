import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("🌲 MODEL 2: Random Forest")
print("=" * 50)

# Load final dataset (with aggregated features)
print("\n[1/5] Loading data...")
df = pd.read_csv('final_dataset.csv')

# Remove rows with no tweets
# df = df[df['tweet_count'] > 0]

print(f"✓ Total days: {len(df)}")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")

# Create additional features
print("\n[2/5] Engineering features...")
df['sentiment_strength'] = abs(df['avg_sentiment'])
df['engagement_rate'] = df['avg_followers'] + df['avg_friends']
df['price_volatility'] = df['high'] - df['low']

# Features for prediction
feature_cols = ['avg_sentiment', 'tweet_count', 'sentiment_strength', 
                'engagement_rate', 'price_volatility', 'volume']

# Check which columns exist
available_features = [col for col in feature_cols if col in df.columns]
print(f"✓ Using {len(available_features)} features:")
for col in available_features:
    print(f"   - {col}")

X = df[available_features]
y = df['label']  # 1 = price went up, 0 = price went down

print(f"\n✓ Samples: {len(X)}")
print(f"✓ Up days: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"✓ Down days: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

# Split data (time-series split - no shuffling)
print("\n[3/5] Splitting data (80% train, 20% test)...")
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# Train Random Forest
print("\n[4/5] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✓ Training complete!")

# Predictions
print("\n[5/5] Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("📊 RESULTS - MODEL 2: RANDOM FOREST")
print("=" * 50)
print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance:")
for idx, row in feature_importance_df.iterrows():
    print(f"   {row['feature']:<20} {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='green')
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model2_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model2_feature_importance.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True)
plt.title(f'Random Forest - Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('model2_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model2_confusion_matrix.png")

# Save model
import pickle
with open('model2_random_forest.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: model2_random_forest.pkl")

# Test on recent data
print("\n" + "=" * 50)
print("🧪 Testing on Recent Days")
print("=" * 50)

test_sample = df.iloc[-5:][['date', 'avg_sentiment', 'tweet_count', 'close', 'label']]
X_sample = df.iloc[-5:][available_features]
predictions = model.predict(X_sample)

for idx, (_, row) in enumerate(test_sample.iterrows()):
    actual = "UP ⬆️" if row['label'] == 1 else "DOWN ⬇️"
    pred = "UP ⬆️" if predictions[idx] == 1 else "DOWN ⬇️"
    correct = "✅" if row['label'] == predictions[idx] else "❌"
    
    print(f"\n📅 {row['date']}")
    print(f"   Sentiment: {row['avg_sentiment']:.3f} | Tweets: {int(row['tweet_count'])}")
    print(f"   Actual: {actual} | Predicted: {pred} {correct}")

print("\n" + "=" * 50)
print("✅ MODEL 2 COMPLETE!")
print("=" * 50)
print(f"Final Accuracy: {accuracy*100:.2f}%")

if accuracy >= 0.70:
    print("🎉 TARGET ACHIEVED! (≥70%)")
else:
    print(f"⚠️ Need {(0.70-accuracy)*100:.1f}% more for 70% target")
