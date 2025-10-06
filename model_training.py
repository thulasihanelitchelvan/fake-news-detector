import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from pathlib import Path

current_dir = Path(__file__).parent

project_root = current_dir

# Paths
features_train_path = project_root / "data" / "tfidf_train.pkl"
features_test_path  = project_root / "data" / "tfidf_test.pkl"
labels_train_path   = project_root / "data" / "labels_train.pkl"
labels_test_path    = project_root / "data" / "labels_test.pkl"
model_path          = project_root / "data" / "fake_news_model.pkl"

# Load features and labels
X_train = joblib.load(features_train_path)
X_test  = joblib.load(features_test_path)
y_train = joblib.load(labels_train_path)
y_test  = joblib.load(labels_test_path)

print("Train features shape:", X_train.shape)
print("Test features shape :", X_test.shape)
print("Train labels:", len(y_train))
print("Test labels :", len(y_test))

# Model (binary, high-dimensional sparse -> liblinear is reliable)
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)

print(f"✅ Accuracy: {acc:.4f}")
print(f"Precision (REAL=1): {prec:.4f}")
print(f"Recall (REAL=1):    {rec:.4f}")
print(f"F1 (REAL=1):        {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["FAKE(0)", "REAL(1)"]))

# Save the model
joblib.dump(model, model_path)
print(f"✅ Model saved successfully: {model_path}")
