import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

# Load and preprocess dataset
data = pd.read_csv("creditcard.csv")
X = data.drop('Class', axis=1)
y = data['Class']

# Scale 'Time' and 'Amount'
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define fast classifiers
models = {
    "logistic_regression": LogisticRegression(),
    "naive_bayes": GaussianNB(),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "sgd_classifier": SGDClassifier(max_iter=1000, tol=1e-3),
    "decision_tree": DecisionTreeClassifier(max_depth=5)
}

# Train and save each model
for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = roc_auc_score(y_test, y_pred)
    print(f"{name} ROC AUC Score: {score}")
    print(classification_report(y_test, y_pred))
    
    filename = f"{name}.pkl"
    joblib.dump(model, filename)
    print(f"✅ Model saved as {filename}")
