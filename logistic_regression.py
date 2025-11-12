import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("🤖 MODEL 1: Logistic Regression")
print("=" * 50)

# Load cleaned tweets
print("\n[1/5] Loading data...")
tweets = pd.read_csv('tweets_cleaned.csv')

# Prepare features and labels
X = tweets['cleaned_text']
y = tweets['sentiment_label'].map({'positive': 1, 'negative': 0, 'neutral': 0})

print(f"✓ Total samples: {len(X)}")
print(f"✓ Positive: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"✓ Negative/Neutral: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

# Split data
print("\n[2/5] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# Convert text to numbers
print("\n[3/5] Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"✓ Created {X_train_vec.shape[1]} features")

# Train model
print("\n[4/5] Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X_train_vec, y_train)
print("✓ Training complete!")

# Predictions
print("\n[5/5] Evaluating model...")
y_pred = model.predict(X_test_vec)
y_pred_proba = model.predict_proba(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("📊 RESULTS - MODEL 1: LOGISTIC REGRESSION")
print("=" * 50)
print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=['Negative/Neutral', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Logistic Regression - Confusion Matrix\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('model1_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model1_confusion_matrix.png")

# Save model
import pickle
with open('model1_logistic_regression.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)
print("✓ Saved: model1_logistic_regression.pkl")

# Test examples
print("\n" + "=" * 50)
print("🧪 Testing with Sample Tweets")
print("=" * 50)

test_examples = [
    "Bitcoin is going to the moon! 🚀 Best investment!",
    "BTC crashing hard, this is terrible",
    "Bitcoin price stable today, just watching"
]

for tweet in test_examples:
    cleaned = tweet.lower()
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    sentiment = "POSITIVE 😊" if pred == 1 else "NEGATIVE/NEUTRAL 😐"
    confidence = max(proba) * 100
    
    print(f"\n📝 \"{tweet}\"")
    print(f"   → {sentiment} (confidence: {confidence:.1f}%)")

print("\n" + "=" * 50)
print("✅ MODEL 1 COMPLETE!")
print("=" * 50)
print(f"Final Accuracy: {accuracy*100:.2f}%")

if accuracy >= 0.70:
    print("🎉 TARGET ACHIEVED! (≥70%)")
else:
    print(f"⚠️ Need {(0.70-accuracy)*100:.1f}% more for 70% target")

print("\n✅ Next step: python model2_random_forest.py")