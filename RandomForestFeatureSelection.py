from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('alzheimer.csv')
print(df.head(5))

df = df.drop(df[df["Group"] == "Converted"].index)
class_le = LabelEncoder()
y = class_le.fit_transform(df['Group'].values)
df.iloc[:, 0] = y
y = class_le.fit_transform(df['M/F'].values)
df.iloc[:, 1] = y

df = df.dropna(axis=0)

X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)
X_test, X_valid, y_test, y_valid = train_test_split(
    X_remaining, y_remaining, test_size=0.5, random_state=0, stratify=y_remaining)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d %-*s %f" %
          (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')

plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()
plt.show()
