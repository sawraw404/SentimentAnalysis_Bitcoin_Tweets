import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("📊 MODEL COMPARISON DASHBOARD")
print("=" * 50)

# Your actual results
results = {
    'Model': ['Logistic Regression', 'Random Forest', 'LSTM Neural Network'],
    'Accuracy': [0.9195, 0.4286, 0.9578],
    'Precision': [0.92, 0.50, 0.96],
    'Recall': [0.91, 0.47, 0.96],
    'F1-Score': [0.92, 0.43, 0.96],
    'Training Time': ['30 sec', '5 sec', '8 min'],
    'Type': ['Traditional ML', 'Ensemble ML', 'Deep Learning']
}

df_results = pd.DataFrame(results)

print("\n📈 Model Performance Summary:")
print("=" * 50)
print(df_results.to_string(index=False))

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Accuracy Comparison
ax1 = fig.add_subplot(gs[0, :2])
colors = ['#3498db', '#e74c3c', '#9b59b6']
bars = ax1.bar(df_results['Model'], df_results['Accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='70% Target')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, df_results['Accuracy'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 2. Best Model Highlight
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
best_idx = df_results['Accuracy'].idxmax()
best_model = df_results.loc[best_idx, 'Model']
best_acc = df_results.loc[best_idx, 'Accuracy']

ax2.text(0.5, 0.7, '🏆 BEST MODEL', ha='center', fontsize=16, fontweight='bold', color='gold',
         transform=ax2.transAxes)
ax2.text(0.5, 0.5, best_model, ha='center', fontsize=14, fontweight='bold',
         transform=ax2.transAxes)
ax2.text(0.5, 0.3, f'{best_acc:.2%} Accuracy', ha='center', fontsize=12, color='green',
         transform=ax2.transAxes, fontweight='bold')

# 3. Detailed Metrics Comparison
ax3 = fig.add_subplot(gs[1, :])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(df_results['Model']))
width = 0.2

for i, metric in enumerate(metrics):
    offset = width * (i - 1.5)
    ax3.bar(x + offset, df_results[metric], width, label=metric, alpha=0.8)

ax3.set_xlabel('Models', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Detailed Performance Metrics', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(df_results['Model'], rotation=15, ha='right')
ax3.legend(fontsize=10)
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.3)

# 4. Model Characteristics
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
ax4.text(0.5, 0.9, 'Logistic Regression', ha='center', fontsize=11, fontweight='bold',
         transform=ax4.transAxes, color='#3498db')
ax4.text(0.5, 0.7, '✓ Fast training', ha='center', fontsize=9, transform=ax4.transAxes)
ax4.text(0.5, 0.55, '✓ Interpretable', ha='center', fontsize=9, transform=ax4.transAxes)
ax4.text(0.5, 0.4, '✓ Low resource', ha='center', fontsize=9, transform=ax4.transAxes)
ax4.text(0.5, 0.25, f'✓ {df_results.loc[0, "Accuracy"]:.1%} accurate', ha='center', fontsize=9,
         transform=ax4.transAxes, color='green', fontweight='bold')

ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
ax5.text(0.5, 0.9, 'Random Forest', ha='center', fontsize=11, fontweight='bold',
         transform=ax5.transAxes, color='#e74c3c')
ax5.text(0.5, 0.7, '✓ Feature importance', ha='center', fontsize=9, transform=ax5.transAxes)
ax5.text(0.5, 0.55, '✓ Handles non-linear', ha='center', fontsize=9, transform=ax5.transAxes)
ax5.text(0.5, 0.4, '⚠ Needs more data', ha='center', fontsize=9, transform=ax5.transAxes)
ax5.text(0.5, 0.25, f'⚠ {df_results.loc[1, "Accuracy"]:.1%} accurate', ha='center', fontsize=9,
         transform=ax5.transAxes, color='orange', fontweight='bold')

ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
ax6.text(0.5, 0.9, 'LSTM', ha='center', fontsize=11, fontweight='bold',
         transform=ax6.transAxes, color='#9b59b6')
ax6.text(0.5, 0.7, '✓ Deep learning', ha='center', fontsize=9, transform=ax6.transAxes)
ax6.text(0.5, 0.55, '✓ Learns patterns', ha='center', fontsize=9, transform=ax6.transAxes)
ax6.text(0.5, 0.4, '✓ Best performance', ha='center', fontsize=9, transform=ax6.transAxes)
ax6.text(0.5, 0.25, f'✓ {df_results.loc[2, "Accuracy"]:.1%} accurate', ha='center', fontsize=9,
         transform=ax6.transAxes, color='green', fontweight='bold')

plt.suptitle('Bitcoin Sentiment Analysis - Model Comparison', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_comparison_dashboard.png")

# Summary statistics
print("\n" + "=" * 50)
print("📊 FINAL SUMMARY")
print("=" * 50)

models_above_70 = (df_results['Accuracy'] >= 0.70).sum()
print(f"✅ Models with ≥70% accuracy: {models_above_70}/3")
print(f"✅ Best model: {best_model} ({best_acc:.2%})")
print(f"✅ Average accuracy: {df_results['Accuracy'].mean():.2%}")

print("\n🎯 PROJECT REQUIREMENTS:")
print("  ✅ 3 different models implemented")
print("  ✅ Sentiment analysis on tweets")
print("  ✅ Bitcoin price prediction")
if models_above_70 >= 1:
    print("  ✅ At least one model ≥70% accuracy")
else:
    print("  ⚠️ Need at least one model ≥70%")

print("\n" + "=" * 50)
print("🎉 PROJECT COMPLETE!")
print("=" * 50)

print("\n📁 Generated Files:")
files = [
    "bitcoin_tweets.csv",
    "bitcoin_prices.csv",
    "tweets_cleaned.csv",
    "final_dataset.csv",
    "model1_logistic_regression.pkl",
    "model1_confusion_matrix.png",
    "model2_random_forest.pkl",
    "model2_confusion_matrix.png",
    "model2_feature_importance.png",
    "model3_lstm.h5",
    "model3_confusion_matrix.png",
    "model3_training_history.png",
    "model_comparison_dashboard.png"
]

for f in files:
    print(f"  ✓ {f}")


