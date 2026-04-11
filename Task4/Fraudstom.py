# Install if needed: pip install imbalanced-learn scikit-learn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# ── 1. Create imbalanced dataset (or replace with your own CSV) ──
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.95, 0.05],   # 95% majority, 5% minority → imbalanced
    random_state=42
)

print("Before SMOTE:")
print(f"  Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")

# ── 2. Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Apply SMOTE ──
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(f"  Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)}")

# ── 4. Train model ──
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# ── 5. Evaluate ──
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
