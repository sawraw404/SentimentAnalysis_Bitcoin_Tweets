import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("🧠 MODEL 3: LSTM Neural Network")
print("=" * 50)

# Load data
print("\n[1/6] Loading data...")
tweets = pd.read_csv('tweets_cleaned.csv')
tweets = tweets.dropna(subset=['cleaned_text', 'sentiment_label'])

X = tweets['cleaned_text'].values
y = tweets['sentiment_label'].map({'positive': 1, 'negative': 0, 'neutral': 0}).values

print(f"✓ Total samples: {len(X)}")
print(f"✓ Positive: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

# Split
print("\n[2/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenization
print("\n[3/6] Tokenizing text...")
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

print(f"✓ Vocab size: {len(tokenizer.word_index)}")
print(f"✓ Sequence length: {max_len}")

# Build model
print("\n[4/6] Building LSTM model...")
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Train
print("\n[5/6] Training LSTM (this may take 5-10 minutes)...")
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate
print("\n[6/6] Evaluating...")
y_pred_proba = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("📊 RESULTS - MODEL 3: LSTM")
print("=" * 50)
print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative/Neutral', 'Positive']))

# Training history plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('model3_training_history.png', dpi=300)
print("\n✓ Saved: model3_training_history.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title(f'LSTM\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('model3_confusion_matrix.png', dpi=300)
print("✓ Saved: model3_confusion_matrix.png")

# Save
model.save('model3_lstm.h5')
print("✓ Saved: model3_lstm.h5")

import pickle
with open('model3_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Saved: model3_tokenizer.pkl")

print("\n" + "=" * 50)
print("✅ MODEL 3 COMPLETE!")
print("=" * 50)
print(f"Accuracy: {accuracy*100:.2f}%")

if accuracy >= 0.70:
    print("🎉 TARGET ACHIEVED!")
else:
    print(f"⚠️ Need {(0.70-accuracy)*100:.1f}% more")

print("\n✅ All 3 models complete!")
print("Next: python compare_models.py")