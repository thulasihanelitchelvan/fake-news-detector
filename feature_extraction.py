import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pathlib import Path

current_dir = Path(__file__).parent

project_root = current_dir

# Paths
cleaned_path = project_root / "data" / "cleaned_data.csv"
features_train_path = project_root / "data" / "tfidf_train.pkl"
features_test_path = project_root / "data" / "tfidf_test.pkl"
labels_train_path = project_root / "data" / "labels_train.pkl"
labels_test_path = project_root / "data" / "labels_test.pkl"
vectorizer_path = project_root / "data" / "tfidf_vectorizer.pkl"

def feature_extraction():
    # Load cleaned dataset
    df = pd.read_csv(cleaned_path, encoding="utf-8")

    # Ensure no NaN
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    # Split BEFORE vectorization to avoid leakage
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF (fit only on training data)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=False  # text is already cleaned/lowercased in preprocessing
    )
    X_train = vectorizer.fit_transform(X_train_text.tolist())
    X_test = vectorizer.transform(X_test_text.tolist())

    # Save features + labels + vectorizer (joblib handles large sparse matrices well)
    joblib.dump(X_train, features_train_path)
    joblib.dump(X_test, features_test_path)
    joblib.dump(y_train.to_numpy(), labels_train_path)
    joblib.dump(y_test.to_numpy(), labels_test_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"✅ TF-IDF train features saved to {features_train_path}  shape={X_train.shape}")
    print(f"✅ TF-IDF test features saved to {features_test_path}   shape={X_test.shape}")
    print(f"✅ Labels saved to {labels_train_path} / {labels_test_path}")
    print(f"✅ Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    feature_extraction()
