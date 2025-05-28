from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess data
df = pd.read_csv("data/heart_cleveland_upload.csv")
X = df.drop("target", axis=1)
y = df["target"]

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = [
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("rf", RandomForestClassifier(class_weight="balanced")),
    ("gb", GradientBoostingClassifier()),
    ("xgb", XGBClassifier(eval_metric='logloss'))
]

ensemble = VotingClassifier(estimators=models, voting="soft")
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(ensemble, "model/model.joblib")
joblib.dump(scaler, "model/scaler.joblib")
