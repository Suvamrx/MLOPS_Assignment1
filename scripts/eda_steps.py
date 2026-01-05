import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('data/raw_heart_disease.csv')

# 1. Clean Data: Handle missing values (Requirement 1) [cite: 15]
# Replace '?' with NaN and fill with median
df = df.fillna(df.median())

# 2. Binary Target: Convert target to 0 (absent) and 1 (present) [cite: 10]
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 3. Visualizations (Requirement 1.3) [cite: 16]
os.makedirs('screenshots', exist_ok=True)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heart Disease Feature Correlation")
plt.savefig('screenshots/heatmap.png')

# Class Balance
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title("Class Balance (0: No Disease, 1: Disease)")
plt.savefig('screenshots/class_balance.png')

# Feature Histograms
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.savefig('screenshots/histograms.png')

print("EDA completed. Check the 'screenshots' folder for your report images.")